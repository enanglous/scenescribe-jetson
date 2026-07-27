# SceneScribe: AI-Enabled Intelligent Glasses for the Visually Impaired

![NVIDIA Jetson](https://img.shields.io/badge/Hardware-NVIDIA%20Jetson%20Orin%20Nano-76B900?style=flat-square&logo=nvidia&logoColor=white)
![Python](https://img.shields.io/badge/Language-Python%203.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![TensorRT](https://img.shields.io/badge/Optimization-NVIDIA%20TensorRT-76B900?style=flat-square&logo=nvidia&logoColor=white)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)


SceneScribe is a fully decentralized, edge-computed wearable solution engineered to provide real-time navigation, obstacle avoidance, and ambient environmental awareness for Visually Impaired People (VIP). Moving away from cloud-dependent architectures, this iteration runs entirely offline on an NVIDIA Jetson Orin Nano, integrating deep learning vision streams, precise fiducial marker tracking, and a hands-free local voice pipeline to maximize user safety, data privacy, and operational reliability.

---

## 📖 Complete Project Documentation

The complete, comprehensive technical documentation, design schematics, implementation logs, and theoretical evaluations are maintained in a separate private repository.
👉 **[Access the Private SceneScribe Documentation Repository](https://github.com/enanglous/scenescribe-docs)** *(Repository access required)*

---

## 📝 To-Do List

- [x] ~~Update `README.md`~~
- [ ] ~~Update Documentation~~
- [ ] Refactor Code
- [ ] Repository Cleanup

## 🏗️ Architecture Overview

The system processes real-world data across two parallel architectures deployed entirely on-device:

### 1. Embedded Hardware Platform

* **Compute Core:** NVIDIA Jetson Orin Nano (8GB) configured in *MAXN SUPER Mode* (15W power cap), delivering 40 TOPS of INT8 local AI performance and 68 GB/s memory bandwidth.
* **Ergonomic Frame:** A custom 3D-printed wearable chassis engineered using two distinct filaments: Rigid **PET-G** for the lens housing to minimize vibrations and prevent camera motion blur, and Flexible **TPU** for the temples to optimize prolonged physical comfort.
* **Audio & Haptic Feedback:** Audio output relies on a low-noise **PAM8403 Class-D amplifier** paired with dual 8-ohm speakers. Immediate mechanical alerts are handled by discrete coin vibration motors driven safely via a **TIP41 NPN power transistor** circuit to safeguard the Jetson’s low-current GPIO pins.

### 2. Software & Deep Learning Pipelines

* **Hands-Free Voice User Interface (VUI):** Combines **Vosk** for local wake-word monitoring ("*Hey Glasses*"), **Faster Whisper** (via CTranslate2 optimized execution) for near-instant offline Speech-to-Text translation, and **Piper** for low-latency neural Text-to-Speech synthesis.
* **Vision-Language Core:** Executes **Qwen-VL-Chat (2B / 3B K-Quantized)** locally via an Ollama wrapper. The model was selected via Multi-Criteria Decision Analysis (Simple Additive Weighting) after benchmark evaluation on the VizWiz-VQA dataset for semantic accuracy (CLIPScore) and hallucination resistance (POPE).
* **Global Navigation:** Utilizes **Dijkstra’s Algorithm** mapped across a topological indoor graph. Localization tracks real-time **ArUco fiducial markers** calculated via Perspective-n-Point (PnP) pose estimation, with a **TensorRT-optimized YOLO classification model** (`yolo26n-cls`) serving as a fallback.
* **Obstacle Avoidance:** A hardware-agnostic, dual-stream vision pipeline that maps instance segmentation (`yolo26n-seg`) onto monocular depth estimation maps (**Depth Anything V2**) running at a continuous edge inference rate of **13.4 FPS**.

---

## 📊 Performance Benchmarks & Metrics

Derived from extensive testing within the Mechatronics Department at NUST CEME:

* **Inference Speeds:** YOLO scene classification runs at ~13 ms (~75 FPS) across 10,422 test frames, achieving **98% overall accuracy**.
* **Localization Precision:** ArUco tracking preserves a spatial distance error of **< 0.005 meters** up to a distance of 1.5 meters, remaining under 0.025 meters at a maximum 2.5-meter range.
* **Latency Reduction:** End-to-end processing delays dropped to **20–25 seconds** (a 30% reduction compared to previous cloud-reliant architectures), while raw Speech-to-Text latency sits at a remarkable **0.67 seconds**.
* **Power & Runtime:** Powered by a 74Wh power bank via a USB-C to DC barrel adapter, providing a stable continuous mobile runtime of **4.2 hours**.

---

## 🛠️ Building and Running Instructions

Follow these instructions to configure the Linux environment on the NVIDIA Jetson Orin Nano, initialize the quantized model configurations, and execute the runtime system.

### 1. Prerequisites & JetPack Setup

Ensure your Jetson Orin Nano is flashed with **NVIDIA JetPack 6.x** (Ubuntu 22.04 LTS based) to ensure native support for CUDA 12.x and TensorRT 10.x.

```bash
# Update and install system dependencies for audio, rendering, and vector building
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y python3-pip portaudio19-dev libatlas-base-dev libgstreamer1.0-dev ffmpeg git

```

### 2. Environment Configuration & Dependencies

1. Clone this repository to your Jetson home directory and navigate to the root:
```bash
git clone https://github.com/enanglous/scenescribe-jetson.git scenescribe
cd scenescribe

```


2. Set the hardware clock to maximum performance configuration:
```bash
sudo nvpmodel -m 0  # Activates MAXN Mode
sudo jetson_clocks  # Locks clocks at maximum frequency for consistent inference rates

```


3. Install Python virtual environment wrapper and download software packages:
```bash
python3 -m venv venv
source venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt

```

> [!WARNING]
> **Disclaimer:** The `requirements.txt` installs OpenCV-Python, Faster-Whisper, Vosk, Piper-TTS, and Ollama core integrations, it does NOT install any required models and many of the core dependancies like the tensorRT engine (you can install those for your system by looking it up it is a fairly simple process). I am too lazy to provide a download link for the models at the moment so please contact me if you need access to those models. Otherwise you can try to put your models into the `./models/` directory but until the code is cleaned up I would not recommend you waste time debugging where they need to go.. the codebase is currently a mess and I plan to at some point fix that)

### 3. Model Engine Optimization & Weights

1. **Initialize the local Ollama VLM instance:**
```bash
# Run the Ollama installation script
curl -fsSL https://ollama.com/install.sh | sh

# Pull the target quantized vision-language model
ollama pull qwen2.5-vl:3b  # Or your specified custom GGUF quantization

```


2. **Compile the TensorRT Engines for YOLO Segmentation/Classification:**
Ensure your model exports (.onnx format) are located in the `/models` directory, then run the TensorRT conversion scripts to maximize hardware utilization:
```bash
/usr/src/tensorrt/bin/trtexec --onnx=models/yolo26n-seg.onnx --saveEngine=models/yolo26n-seg.engine --fp16
/usr/src/tensorrt/bin/trtexec --onnx=models/yolo26n-cls.onnx --saveEngine=models/yolo26n-cls.engine --fp16

```



### 4. Running the Main Orchestrator

Verify that the CSI camera module or USB camera array is correctly attached, and that the PAM8403 audio out and TIP41 haptic transistor rails are tied to pin **32** and **33** on the microcontroller.

Execute the master multi-threaded runtime loops:

```bash
python -m src.flask_backend.websockets_backend

```

### 5. Operating System Loop

Once initialized, the system runs hands-free:

1. **Wake Word Active:** Speak "*Hey Glasses*" aloud. The local `Vosk` thread triggers an auditory response beep through the speakers.
2. **Command Processing:** Speak your target intent (e.g., "*Where am I?*" or "*Describe what is on the table*").
3. **Execution Routing:** * If a navigation query is parsed, the system activates the ArUco PnP pose pipeline paired with Dijkstra calculation loops.
* If an environmental description query is parsed, the local VLM pipeline samples a high-resolution frame and provides an offline auditory overview.
* The underlying safety stream fusions instance segmentation with depth maps at **13.4 FPS**, triggering the TIP41 haptic vibration array autonomously if obstacles present a collision hazard.
