import os
import torch
import numpy as np
import cv2
import tensorrt as trt

class DepthEstimator:
    """
    Depth estimation using Depth Anything v2 (TensorRT version)
    """
    def __init__(self, engine_path, device='cuda'):
        """
        Initialize the TensorRT depth estimator

        Args:
            engine_path (str): Path to the TensorRT .engine file
            device (str): Device to run inference on (must be 'cuda' for TensorRT)
        """
        if device != 'cuda' or not torch.cuda.is_available():
            raise ValueError("TensorRT requires a CUDA-compatible GPU.")

        self.device = device
        print(f"Loading TensorRT engine from {engine_path}...")

        # 1. Initialize TensorRT Logger and Runtime
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        # 2. Load and deserialize the engine
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        # 3. Create execution context
        self.context = self.engine.create_execution_context()

        # 4. Get input and output tensor names (TRT 10 API)
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)

        # 5. Extract expected input shape (e.g., [1, 3, 518, 518])
        self.input_shape = self.engine.get_tensor_shape(self.input_name)
        if self.input_shape[0] == -1: # Handle dynamic batch sizes
            self.input_shape = (1, *self.input_shape[1:])
            self.context.set_input_shape(self.input_name, self.input_shape)

        self.output_shape = self.engine.get_tensor_shape(self.output_name)
        if self.output_shape[0] == -1:
            self.output_shape = (1, *self.output_shape[1:])

        # 6. Allocate output memory on the GPU using PyTorch
        self.output_tensor = torch.empty(
            tuple(self.output_shape),
            dtype=torch.float32,
            device=self.device
        )

        self.stream = torch.cuda.Stream()

        print("TensorRT Engine loaded successfully.")

    def estimate_depth(self, image):
        """
        Estimate depth from an image using TensorRT

        Args:
            image (numpy.ndarray): Input image (BGR format)

        Returns:
            numpy.ndarray: Depth map (normalized to 0-1)
        """
        orig_h, orig_w = image.shape[:2]
        input_h, input_w = self.input_shape[2:] # Usually 518x518 for Depth Anything

        # --- PREPROCESSING ---
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to engine's expected input size
        img_resized = cv2.resize(img_rgb, (input_w, input_h), interpolation=cv2.INTER_CUBIC)

        # Normalize (Depth Anything standard ImageNet normalization)
        img_normalized = img_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_normalized = (img_normalized - mean) / std

        # Change format from HWC to CHW and add batch dimension
        img_chw = img_normalized.transpose(2, 0, 1)
        img_batched = np.expand_dims(img_chw, axis=0)

        # Move input data to GPU
        input_tensor = torch.from_numpy(img_batched).contiguous().to(self.device)

        # --- INFERENCE ---
        # Assign memory pointers to TensorRT (TRT 10 API)
        self.context.set_tensor_address(self.input_name, input_tensor.data_ptr())
        self.context.set_tensor_address(self.output_name, self.output_tensor.data_ptr())

        # Execute async on the current PyTorch CUDA stream
        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)

        # Wait for the GPU to finish
        self.stream.synchronize()

        # --- POSTPROCESSING ---
        # Pull output back to CPU
        depth_map = self.output_tensor.cpu().numpy().squeeze()

        # Resize back to original image dimensions
        depth_map = cv2.resize(depth_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        ## Normalize depth map to 0-1
        #depth_min = depth_map.min()
        #depth_max = depth_map.max()
        #if depth_max > depth_min:
        #    depth_map = (depth_map - depth_min) / (depth_max - depth_min)

        return depth_map

    def colorize_depth(self, depth_map, cmap=cv2.COLORMAP_INFERNO):
        """Colorize depth map for visualization"""
        d_min = depth_map.min()
        d_max = depth_map.max()

        if d_max > d_min:
            norm_map = (depth_map - d_min) / (d_max - d_min)
        else:
            norm_map = depth_map

        depth_map_uint8 = (norm_map * 255).astype(np.uint8)
        colored_depth = cv2.applyColorMap(depth_map_uint8, cmap)
        return colored_depth

    def get_depth_at_point(self, depth_map, x, y):
        """Get depth value at a specific point"""
        if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
            return depth_map[y, x]
        return 0.0

    def get_depth_in_region(self, depth_map, bbox, method='median'):
        """Get depth value in a region defined by a bounding box"""
        x1, y1, x2, y2 = [int(coord) for coord in bbox]

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(depth_map.shape[1] - 1, x2), min(depth_map.shape[0] - 1, y2)

        region = depth_map[y1:y2, x1:x2]
        if region.size == 0:
            return 0.0

        if method == 'mean': return float(np.mean(region))
        elif method == 'min': return float(np.min(region))
        else: return float(np.median(region))
