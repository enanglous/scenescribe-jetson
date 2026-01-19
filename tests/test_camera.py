import cv2

def capture_image_final(device_id=0, width=1920, height=1080):
    # We add a specific 'videoconvert' to bridge NVIDIA memory to OpenCV
    gst_pipeline = (
        f"v4l2src device=/dev/video{device_id} ! "
        f"image/jpeg, width={width}, height={height}, framerate=30/1 ! "
        "nvv4l2decoder mjpeg=1 ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink drop=True"
    )

    print("Opening hardware pipeline...")
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    # cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    if not cap.isOpened():
        print("Error: OpenCV still cannot bridge the GStreamer pipeline.")
        return

    try:
        # Warm-up (important for MJPEG exposure calibration)
        # print("Warming up...")
        # for _ in range(15):
        #     cap.read()
            
        ret, frame = cap.read()
        if ret:
            cv2.imwrite("jetson_capture.jpg", frame)
            print("Successfully captured jetson_capture.jpg using Hardware Acceleration!")
        else:
            print("Error: Could not retrieve frame from pipeline.")
            
    finally:
        cap.release()

if __name__ == "__main__":
    capture_image_final()