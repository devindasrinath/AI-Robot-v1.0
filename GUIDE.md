# Nexu Pet Robot — Complete Build Guide

**Robot name:** Nexu  
**Personality:** Unique cute being — not a cat, not a dog. Curious, playful, affectionate but independent.  
**Platform:** SG2002 (Milk-V Duo S) + ESP32-C3  
**Philosophy:** All AI runs on-device. No cloud. No LLM. Real pet-like behavior through ML.

---

## Table of Contents

1. [Project Architecture](#1-project-architecture)
2. [Hardware](#2-hardware)
3. [Environment Setup](#3-environment-setup)
4. [Board Access](#4-board-access)
5. [Phase 1 — Wake Word Model (DS-CNN)](#5-phase-1--wake-word-model-ds-cnn)
6. [Phase 2 — GRU Behavior Policy](#6-phase-2--gru-behavior-policy)
7. [tpu-mlir Conversion Rules](#7-tpu-mlir-conversion-rules-lessons-learned)
8. [Phase 3 — Face Detection](#8-phase-3--face-detection-✅-complete)
9. [Phase 4 — ESP32-C3 Firmware](#9-phase-4--esp32-c3-firmware-✅-complete)
10. [Phase 5 — SG2002 Main Loop](#10-phase-5--sg2002-main-loop-✅-complete)
11. [Phase 6 — Integration (Pending)](#11-phase-6--integration-pending)

---

## 1. Project Architecture

```
┌─────────────────────────────────────────┐     UART 921600 baud
│              SG2002 (Linux)              │ ◄──────────────────────► ESP32-C3
│                                          │
│  ┌──────────┐   ┌────────────────────┐  │     ESP32-C3 handles:
│  │ GC2083   │   │  USB Microphone    │  │     - TFT SPI display
│  │ Camera   │   │  (ALSA hw:1,0)     │  │     - I2S speaker (MAX98357A)
│  └──┬───────┘   └────────┬───────────┘  │     - Capacitive touch (TTP223)
│     │                    │              │     - Motor PWM (L298N/TB6612)
│  [NPU] Face Detect    [CPU] Audio       │     - Obstacle sensors (future)
│  [NPU] Wake Word CNN  Preprocessing     │
│  [NPU] GRU Behavior Policy              │
│                                          │
│  Main loop: perceive → decide → act      │
└─────────────────────────────────────────┘
```

### ML Models Summary

| Model | Input | Output | Size | NPU Latency |
|-------|-------|--------|------|-------------|
| Wake Word DS-CNN | [1,1,40,98] mel spectrogram | [1,2] logits | 25KB | 182µs |
| GRU Behavior Policy | [1,1,16] obs + [1,64] hidden | [1,12] logits + [1,64] hidden | 39KB | 125µs |
| Face Detection (SDK) | [1,3,768,432] image | bboxes | ~2MB | ~10ms |

**Total custom inference budget: ~300µs per cycle.**

---

## 2. Hardware

### SG2002 (Milk-V Duo S)
- CPU: RISC-V + ARM Cortex-A53 (runs Linux)
- NPU: CV181x, ~1 TOPS INT8
- RAM: 512MB
- I/O: USB-A, CSI camera, GPIO, UART

### ESP32-C3
- Handles all peripherals to offload SG2002
- Communicates via UART at 921600 baud

### Peripheral Wiring Plan

| Peripheral | Component | Connected To | Notes |
|------------|-----------|--------------|-------|
| Camera | GC2083 | SG2002 CSI | Officially supported by SDK |
| Display | ILI9341 320×240, parallel 8080 interface | ESP32-C3 GPIO 0-10 | D0-D7=GPIO0-7, WR=8, CS=9, DC=10 |
| Microphone | USB mic | SG2002 USB-A | ALSA `hw:1,0` — ordered, arriving ~2 days |
| Speaker | PCM5102A I2S DAC + PAM8403 Class-D amp + 8Ω | ESP32-C3 I2S | Ordered, arriving ~2 days. SG2002 sends PLAY:name over UART |
| Head touch | TTP223 capacitive | ESP32-C3 GPIO2 | ESP32 sends TOUCH:head over UART |
| Motors | 2x encoder motors | ESP32-C3 PWM | Skipped in initial phase |
| Obstacle | HC-SR04 or VL53L0X | ESP32-C3 GPIO/I2C | Not decided yet |

### UART Protocol (SG2002 → ESP32-C3)
```
DISPLAY:happy\n       → show happy face expression
DISPLAY:curious\n     → show curious face
DISPLAY:alert\n
DISPLAY:sleep\n
DISPLAY:idle\n
PLAY:vocal_happy.wav\n    → play sound (ESP32 strips .wav, matches synth by name)
PLAY:vocal_curious.wav\n
PLAY:vocal_alert.wav\n
PLAY:vocal_play.wav\n
MOVE:fwd,60\n         → motor command (future)
```

### UART Protocol (ESP32-C3 → SG2002)
```
TOUCH:head\n         → head touch sensor triggered
TOUCH:body\n
COLLISION:front\n    → obstacle detected
ENCODER:100,98\n     → wheel encoder ticks (future)
```

---

## 3. Environment Setup

### 3.1 Conda Training Environment

```bash
conda create -n pet_robot python=3.12 -y
conda activate pet_robot
pip install torch torchvision torchaudio==2.10.0
pip install numpy librosa soundfile tqdm gtts onnx onnxruntime filelock
```

> **Note:** torchaudio must be pinned to 2.10.0 to match torch 2.10.0. Mismatch causes `_torchaudio.abi3.so` load failure.

### 3.2 Docker for tpu-mlir Conversion

Pull the Sophgo tpu-mlir Docker image:
```bash
docker pull sophgo/tpuc_dev:latest
```

The tpu-mlir wheel (247MB) is downloaded separately to avoid Docker timeout issues:
```bash
mkdir -p /tmp/tpu_wheels
# Download with wget (more reliable than pip for large files):
wget -c -O /tmp/tpu_wheels/tpu_mlir-1.27-py3-none-any.whl \
  https://github.com/sophgo/tpu-mlir/releases/download/v1.27/tpu_mlir-1.27-py3-none-any.whl
# Verify:
unzip -t /tmp/tpu_wheels/tpu_mlir-1.27-py3-none-any.whl | tail -3
```

> **Note:** Always run Docker with `--network host` to avoid DNS resolution failures inside the container.

### 3.3 Directory Structure

```
pet_robot/
├── CLAUDE.md                    ← read by Claude Code automatically
├── GUIDE.md                     ← this file
├── requirements_training.txt
├── wake_word_model/
│   ├── model/
│   │   ├── wake_word_cnn.py     DS-CNN architecture
│   │   ├── data_pipeline.py     TTS + GSC dataset builder
│   │   └── train.py             Training script
│   ├── export/
│   │   ├── export_onnx.py
│   │   └── wake_word.onnx
│   ├── convert/
│   │   └── convert.sh           tpu-mlir Docker conversion
│   ├── data/                    X_train.npy, y_train.npy, X_val.npy, y_val.npy
│   ├── calibration_data/samples/  100 .npz files for INT8 calibration
│   └── wake_word.cvimodel       ← deployed to board /root/
├── gru_behavior_model/
│   ├── model/
│   │   ├── gru_policy.py        ManualGRUCell architecture
│   │   ├── simulate.py          Behavioral simulation / data generation
│   │   └── train.py             Training script
│   ├── export/
│   │   ├── export_onnx.py
│   │   └── gru_behavior.onnx
│   ├── convert/
│   │   └── convert.sh
│   ├── data/                    X_train.npy, y_train.npy, X_val.npy, y_val.npy
│   ├── calibration_data/samples/  200 .npz files
│   └── gru_behavior.cvimodel    ← deployed to board /root/
├── sg2002/
│   ├── include/
│   │   ├── cviruntime.h
│   │   └── cvitpu_debug.h
│   ├── lib/
│   │   ├── libcviruntime.so
│   │   └── libcvikernel.so
│   ├── wake_word/
│   │   ├── wake_word_test.c
│   │   ├── Makefile
│   │   └── build.sh
│   ├── gru_behavior/
│   │   ├── gru_behavior_test.c
│   │   ├── Makefile
│   │   └── build.sh
│   ├── face_detect/
│   │   ├── face_detector.h/c        SCRFD decoder + NMS
│   │   ├── face_detect_test.c
│   │   ├── Makefile
│   │   └── build.sh
│   └── main_loop/                   ← Phase 5 — SG2002 brain
│       ├── obs.h                    ObsVector struct (16 floats)
│       ├── inference.h/c            loads wake_word + GRU models, NPU mutex
│       ├── uart_sg.h/c              /dev/ttyS1 921600, RX thread, event flags
│       ├── audio.h/c                arecord ring buffer, FFT, mel spectrogram
│       ├── action.h/c               temperature sampling, UART dispatch
│       ├── main.c                   10Hz loop: observe→infer→act
│       ├── Makefile
│       └── build.sh
└── esp32/
    ├── CMakeLists.txt               project(nexu_esp32)
    ├── sdkconfig.defaults           esp32c3, FreeRTOS 1000Hz, SPI DMA
    └── main/
        ├── CMakeLists.txt
        ├── main.c                   FreeRTOS tasks + queue wiring
        ├── display/
        │   ├── st7789.h/c           SPI driver, 40MHz, LEDC backlight
        │   ├── gfx.h/c              fill_rect/circle/ellipse/arc/line
        │   └── face.h/c             animated face: lerp morph, blink, pupil drift, breathing
        ├── uart_comm/
        │   └── uart_comm.h/c        UART1 921600 baud, line protocol, rx/tx tasks
        ├── sensors/
        │   └── touch.h/c            TTP223 GPIO2, 80ms debounce, 500ms cooldown
        └── audio/
            └── i2s_audio.h/c        I2S + PCM5102A, 44100Hz, 4 synthesized vocal sounds
```

---

## 4. Board Access

### SSH
```bash
sshpass -p milkv ssh root@192.168.8.118 "your-command"
```

### File Transfer
SCP is broken on the board (sftp-server missing). Use HTTP + wget instead:

```bash
# On host — serve current directory:
python3 -m http.server 8765 --bind 0.0.0.0 &

# On board (via SSH):
sshpass -p milkv ssh root@192.168.8.118 \
  "wget -q http://192.168.8.124:8765/filename -O /root/filename"

# Kill the server when done:
kill %1
```

> Host IP visible to the board: `192.168.8.124` (check with `ip route get 192.168.8.118`)

### Cross-Compile for Board (aarch64)
Use Debian 9 Docker (has `gcc-aarch64-linux-gnu`):

```bash
docker run --rm --network host \
  -v "/home/devinda_121260/Music/AI projects/pet_robot":/workspace \
  debian:9 bash /workspace/sg2002/<module>/build.sh
```

Each module has a `build.sh` that installs the cross-compiler and runs `make`.

---

## 5. Phase 1 — Wake Word Model (DS-CNN)

**Goal:** Detect "Nexu" from 1-second audio windows, running on NPU.

### 5.1 Architecture — `wake_word_model/model/wake_word_cnn.py`

**DS-CNN (Depthwise Separable CNN)** — industry standard for embedded keyword spotting.

```
Input: [batch, 1, 40, 98]   ← 1ch, 40 mel bins, 98 frames (~1 second)

Stem:    Conv2d(1→32, k=3) + BN + ReLU
DS Block 1: DW(32) + PW(32→32, stride=2) + BN + ReLU  → [32, 20, 49]
DS Block 2: DW(32) + PW(32→64, stride=2) + BN + ReLU  → [64, 10, 25]
DS Block 3: DW(64) + PW(64→64, stride=2) + BN + ReLU  → [64,  5, 13]
DS Block 4: DW(64) + PW(64→64, stride=1) + BN + ReLU  → [64,  5, 13]
AvgPool2d(kernel_size=(5, 13))                          → [64,  1,  1]
Flatten                                                 → [64]
Linear(64 → 2)                                          → [no_wake, wake]

Parameters: ~13,922
```

> **CRITICAL:** Use `nn.AvgPool2d(kernel_size=(5, 13))` — NOT `nn.AdaptiveAvgPool2d(1)`.  
> AdaptiveAvgPool2d exports as `ReduceMean` in ONNX which tpu-mlir cannot compile.

```python
# WRONG — will fail in tpu-mlir:
self.gap = nn.AdaptiveAvgPool2d(1)

# CORRECT:
self.gap = nn.AvgPool2d(kernel_size=(5, 13))  # matches exact feature map [64, 5, 13]
```

### 5.2 Data Pipeline — `wake_word_model/model/data_pipeline.py`

**Positive samples ("Nexu" recordings):**
- Generated with gTTS (Google TTS, online) and espeak-ng (offline fallback)
- 7 phrase variants × 3 accents = ~21 base samples
- Augmented with: time-stretch (4 rates), pitch-shift (4 steps), background noise (3 mixes), volume jitter (4 levels)
- **Total: 294 positive samples**

**Negative samples (background speech, not "Nexu"):**
- Google Speech Commands v0.02 dataset: words like yes, no, up, down, stop, go, etc.
- 20 words × 80 samples = **1600 negative samples**

**Mel spectrogram extraction:**
```python
SAMPLE_RATE   = 16000
N_MELS        = 40
N_FFT         = 512       # 32ms window
HOP_LENGTH    = 160       # 10ms hop
TARGET_FRAMES = 98        # ~1 second
```

**Run data pipeline:**
```bash
cd "/home/devinda_121260/Music/AI projects/pet_robot"
conda run -n pet_robot python wake_word_model/model/data_pipeline.py
```

**What it produces:**
```
wake_word_model/data/X_train.npy   [N, 1, 40, 98]  float32
wake_word_model/data/y_train.npy   [N]              int64  (0=no_wake, 1=wake)
wake_word_model/data/X_val.npy
wake_word_model/data/y_val.npy
wake_word_model/calibration_data/samples/sample_0000.npz ... sample_0099.npz
```

**Known issues and fixes:**
- `torchaudio` dataset loader requires TorchCodec — **fix:** load wav files directly with librosa
- GSC download needs the directory to exist first:
  ```bash
  mkdir -p wake_word_model/data/speech_commands/
  ```
- GSC extracted path: `data/speech_commands/SpeechCommands/speech_commands_v0.02/`

### 5.3 Training — `wake_word_model/model/train.py`

**Key training decisions:**
- `WeightedRandomSampler` — handles 1600:294 class imbalance
- `SpecAugment` — frequency masking (up to 8 bins) + time masking (up to 15 frames)
- `CrossEntropyLoss`, Adam lr=1e-3, CosineAnnealingLR, 40 epochs
- **Save best model by recall** (not accuracy) — we want zero missed wake words

```bash
conda run -n pet_robot python wake_word_model/model/train.py
```

**Results:**
```
Best recall: 1.000  (no wake words missed)
FPR:         0.000  (no false triggers)
Val set:     190 samples
```

Saved to: `wake_word_model/model/wake_word.pth`

### 5.4 ONNX Export — `wake_word_model/export/export_onnx.py`

```bash
conda run -n pet_robot python wake_word_model/export/export_onnx.py
```

**Critical export flags:**
```python
torch.onnx.export(
    model, dummy, ONNX_PATH,
    opset_version=13,
    input_names=["mel_input"],
    output_names=["logits"],
    dynamic_axes=None,   # static shapes required — tpu-mlir rejects dynamic
    dynamo=False,        # REQUIRED — new exporter can't produce opset 13
)
```

**Verify the ONNX has no ReduceMean:**
```bash
conda run -n pet_robot python -c "
import onnx
m = onnx.load('wake_word_model/export/wake_word.onnx')
ops = set(n.op_type for n in m.graph.node)
print('Ops:', ops)
print('Opset:', m.opset_import[0].version)
assert 'ReduceMean' not in ops, 'FAIL: ReduceMean found!'
assert 'AveragePool' in ops, 'FAIL: AveragePool missing!'
print('OK')
"
```

Expected output: `Ops: {'Relu', 'Gemm', 'Constant', 'Conv', 'Reshape', 'AveragePool'}  Opset: 13`

### 5.5 tpu-mlir Conversion — `wake_word_model/convert/convert.sh`

```bash
docker run --rm --network host \
  -v "/home/devinda_121260/Music/AI projects/pet_robot/wake_word_model":/workspace \
  -v "/tmp/tpu_wheels":/wheels \
  sophgo/tpuc_dev:latest \
  bash /workspace/convert/convert.sh
```

**What `convert.sh` does:**
1. Installs tpu-mlir from local wheel (or PyPI fallback)
2. `model_transform.py` — ONNX → MLIR (input_shapes `[[1,1,40,98]]`)
3. `run_calibration.py` — generates INT8 calibration table from 100 sample .npz files
4. `model_deploy.py` — MLIR → cvimodel (INT8 quantized, chip=cv181x)
5. GenericCpu check — only `quant` ops should appear (boundary quantization, not CPU fallback)

**Output:** `wake_word_model/wake_word.cvimodel` (25KB)

**Expected final output:**
```
Done! Output: /workspace/wake_word.cvimodel
```

**GenericCpu check — what's OK:**
```
%2 = "tpu.GenericCpu"(%1) {cpu_op_name = "quant", ...}
WARNING: CPU ops found — check names above
```
This `quant` op is just the input INT8 boundary — it is expected and fine.

### 5.6 Deploy and Verify on Board

**Transfer cvimodel:**
```bash
cd "/home/devinda_121260/Music/AI projects/pet_robot/wake_word_model" && \
python3 -m http.server 8765 --bind 0.0.0.0 &
sshpass -p milkv ssh root@192.168.8.118 \
  "wget -q http://192.168.8.124:8765/wake_word.cvimodel -O /root/wake_word.cvimodel"
kill %1
```

**Build test binary:**
```bash
docker run --rm --network host \
  -v "/home/devinda_121260/Music/AI projects/pet_robot":/workspace \
  debian:9 bash /workspace/sg2002/wake_word/build.sh
```

**Transfer test binary:**
```bash
cd "/home/devinda_121260/Music/AI projects/pet_robot/sg2002/wake_word" && \
python3 -m http.server 8766 --bind 0.0.0.0 &
sshpass -p milkv ssh root@192.168.8.118 \
  "wget -q http://192.168.8.124:8766/wake_word_test -O /root/wake_word_test && chmod +x /root/wake_word_test"
kill %1
```

**Run on board:**
```bash
sshpass -p milkv ssh root@192.168.8.118 "./wake_word_test wake_word.cvimodel"
```

**Expected output:**
```
=== Wake Word NPU Verification ===
Inputs: 1  Outputs: 1
  in[0]  'mel_input'
  out[0] 'logits_Gemm_f32'

Test 1 — zeros (silence):
  logits: [no_wake=2.3750, wake=-3.0156]  → no_wake

Latency benchmark (20 runs):
  min=178us  max=217us  avg=182us

PASS: avg latency 182us < 10ms → running on NPU
```

---

## 6. Phase 2 — GRU Behavior Policy

**Goal:** Decide Nexu's next action every 100ms based on sensor inputs + memory (hidden state).

### 6.1 Architecture — `gru_behavior_model/model/gru_policy.py`

**Why ManualGRUCell instead of nn.GRU:**  
tpu-mlir cannot compile `nn.GRU` or `nn.LSTM`. We implement GRU as explicit linear operations (Sigmoid, Tanh, MatMul) which tpu-mlir handles natively.

```
GRU equations implemented manually:
  r = sigmoid(Wr*x + Ur*h + br)      ← reset gate
  z = sigmoid(Wz*x + Uz*h + bz)      ← update gate
  n = tanh(Wn*x + r*(Un*h + bn))     ← new gate
  h' = (1-z)*n + z*h                 ← new hidden state

Input: obs [batch, 1, OBS_SIZE=16]
       hidden_in [batch, HIDDEN_SIZE=64]
Output: action_logits [batch, NUM_ACTIONS=12]
        hidden_out [batch, HIDDEN_SIZE=64]

Parameters: ~36,000
```

### 6.2 Observation Space (16 features)

All values normalized to [0, 1]:

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | touch_head | Head touch sensor triggered |
| 1 | touch_body | Body touch triggered |
| 2 | touch_tail | Tail touch triggered |
| 3 | collision_front | Front obstacle detected |
| 4 | collision_left | Left obstacle detected |
| 5 | collision_right | Right obstacle detected |
| 6 | person_visible | Camera detects a person |
| 7 | person_distance | 1.0=very close, 0.0=far/absent |
| 8 | wake_word | "Nexu" just detected (one-shot) |
| 9 | battery_level | 1.0=full |
| 10 | time_since_touch | 1.0=been a long time (restless) |
| 11 | idle_ticks_norm | 1.0=been idle very long |
| 12 | sound_level | 1.0=loud environment |
| 13 | motion_energy | 1.0=was moving recently |
| 14 | personality_curiosity | Fixed at boot, 0-1 |
| 15 | personality_playful | Fixed at boot, 0-1 |

### 6.3 Action Space (12 actions)

| Index | Action | Trigger |
|-------|--------|---------|
| 0 | idle_sit | Default quiet state |
| 1 | look_around | Idle/curious |
| 2 | approach | Person seen from distance |
| 3 | follow | Person visible |
| 4 | play_gesture | Wake word / playful mood |
| 5 | vocal_happy | Touched / person close |
| 6 | vocal_curious | Idle, hears sounds |
| 7 | vocal_alert | Tail touched unexpectedly |
| 8 | sleep | Low battery / long idle |
| 9 | avoid_obstacle | Collision detected |
| 10 | self_groom | Idle behavior |
| 11 | wag_tail | Happy / wake word |

### 6.4 Behavior Simulation — `gru_behavior_model/model/simulate.py`

Generates training data using a hand-crafted rule-based policy. Four scenario types:

1. **Person visit** — person walks up, plays, pats Nexu, leaves
2. **Idle exploration** — Nexu alone, grooming/looking around/sleeping
3. **Collision** — Nexu encounters obstacles
4. **Low battery** — Nexu gradually winds down

Nexu's personality (curiosity + playfulness) varies per episode using Gaussian sampling.

```bash
conda run -n pet_robot python gru_behavior_model/model/simulate.py
```

**Results (600 episodes):**
```
Train: 70,807 steps   Val: 7,868 steps
Action distribution: idle_sit 58.5%, sleep 25.1%, look_around 3.6%, ...
Saved 200 calibration samples
```

### 6.5 Training — `gru_behavior_model/model/train.py`

```bash
conda run -n pet_robot python gru_behavior_model/model/train.py
```

**Training setup:**
- `WeightedRandomSampler` — handles severe class imbalance (idle_sit 58%)
- `CrossEntropyLoss`, Adam lr=1e-3, CosineAnnealingLR, 60 epochs
- Gradient clipping at 1.0
- Early stopping (patience=12)

**Results:**
```
Best val_acc: 0.749  (74.9% on 12-class imbalanced problem — good)
Early stopped at epoch 26
```

Saved to: `gru_behavior_model/model/gru_behavior.pth`

### 6.6 ONNX Export — `gru_behavior_model/export/export_onnx.py`

```bash
conda run -n pet_robot python gru_behavior_model/export/export_onnx.py
```

**Verify no GRU/LSTM ops:**
```bash
conda run -n pet_robot python -c "
import onnx
m = onnx.load('gru_behavior_model/export/gru_behavior.onnx')
ops = set(n.op_type for n in m.graph.node)
print('Ops:', ops)
assert 'GRU' not in ops and 'LSTM' not in ops
print('OK — no GRU/LSTM ops')
"
```

Expected ops: `{'Gemm', 'Gather', 'MatMul', 'Constant', 'Mul', 'Add', 'Tanh', 'Sub', 'Sigmoid'}`

### 6.7 tpu-mlir Conversion

```bash
docker run --rm --network host \
  -v "/home/devinda_121260/Music/AI projects/pet_robot/gru_behavior_model":/workspace \
  -v "/tmp/tpu_wheels":/wheels \
  sophgo/tpuc_dev:latest \
  bash /workspace/convert/convert.sh
```

Input shapes: `[[1,1,16],[1,64]]` (obs + hidden_in)  
Output: `gru_behavior_model/gru_behavior.cvimodel` (39KB)

### 6.8 Deploy and Verify on Board

**Transfer cvimodel:**
```bash
cd "/home/devinda_121260/Music/AI projects/pet_robot/gru_behavior_model" && \
python3 -m http.server 8767 --bind 0.0.0.0 &
sshpass -p milkv ssh root@192.168.8.118 \
  "wget -q http://192.168.8.124:8767/gru_behavior.cvimodel -O /root/gru_behavior.cvimodel"
kill %1
```

**Build and transfer test binary:**
```bash
docker run --rm --network host \
  -v "/home/devinda_121260/Music/AI projects/pet_robot":/workspace \
  debian:9 bash /workspace/sg2002/gru_behavior/build.sh

cd "/home/devinda_121260/Music/AI projects/pet_robot/sg2002/gru_behavior" && \
python3 -m http.server 8768 --bind 0.0.0.0 &
sshpass -p milkv ssh root@192.168.8.118 \
  "wget -q http://192.168.8.124:8768/gru_behavior_test -O /root/gru_behavior_test && chmod +x /root/gru_behavior_test"
kill %1
```

**Run on board:**
```bash
sshpass -p milkv ssh root@192.168.8.118 "./gru_behavior_test gru_behavior.cvimodel"
```

**Expected output:**
```
Inputs: 2  Outputs: 2
  in[0]  'obs'
  in[1]  'hidden_in'
  out[0] 'action_logits_Gemm_f32'
  out[1] 'hidden_out_Add_f32'

Running 30 inference steps:
  step  0  idle_sit         (268us)
  step  5  play_gesture     (117us)   ← wake word heard
  step 10  wag_tail         (117us)   ← person appeared
  step 12  wag_tail         (115us)   ← head touched
  step 14  vocal_happy      (116us)
  step 16  approach         (118us)
  step 18  look_around      (116us)   ← person leaving
  step 20  idle_sit         (126us)   ← person gone

Latency: min=114us  max=268us  avg=125us
PASS: avg 125us < 10ms → NPU execution confirmed
```

---

## 7. tpu-mlir Conversion Rules (Lessons Learned)

These are things that **will silently fail or produce wrong output** if you get them wrong.

### Rule 1: No AdaptiveAvgPool2d
```python
# WRONG — exports as ReduceMean (unsupported by tpu-mlir):
self.gap = nn.AdaptiveAvgPool2d(1)

# CORRECT — use exact feature map dimensions:
self.gap = nn.AvgPool2d(kernel_size=(5, 13))   # match [H, W] after your last conv
```

### Rule 2: No nn.GRU / nn.LSTM
```python
# WRONG:
self.gru = nn.GRU(input_size=16, hidden_size=64)

# CORRECT — implement gate equations manually:
class ManualGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        self.Wr = nn.Linear(input_size, hidden_size)
        self.Ur = nn.Linear(hidden_size, hidden_size, bias=False)
        # ... etc
    def forward(self, x, h):
        r = torch.sigmoid(self.Wr(x) + self.Ur(h))
        z = torch.sigmoid(self.Wz(x) + self.Uz(h))
        n = torch.tanh(self.Wn(x) + r * self.Un(h))
        return (1 - z) * n + z * h
```

### Rule 3: Force opset 13 with legacy exporter
```python
# The new PyTorch ONNX exporter (default in PyTorch 2.9+) cannot produce opset 13.
# Must use dynamo=False to force the legacy TorchScript exporter:
torch.onnx.export(
    model, inputs, path,
    opset_version=13,
    dynamo=False,        # ← REQUIRED
    dynamic_axes=None,   # ← static shapes required
)
```

### Rule 4: Static shapes everywhere
- `dynamic_axes=None` in ONNX export
- `--input_shapes "[[1,1,40,98]]"` in model_transform.py
- Never use variable-length sequences or dynamic batch sizes

### Rule 5: GenericCpu quant ops are fine
```
"tpu.GenericCpu" {cpu_op_name = "quant"}   ← OK, expected
"tpu.GenericCpu" {cpu_op_name = "dequant"} ← OK, expected
"tpu.GenericCpu" {cpu_op_name = "softmax"} ← WARNING: CPU fallback!
```
Only `quant` and `dequant` GenericCpu ops are acceptable.

### Rule 6: Docker DNS
```bash
# WRONG — DNS fails inside container:
docker run --rm sophgo/tpuc_dev:latest ...

# CORRECT:
docker run --rm --network host sophgo/tpuc_dev:latest ...
```

### Rule 7: NPU single-context in production
The CV181x NPU can only run one model at a time. In the final C application, all `CVI_NN_Forward` calls must be protected by a mutex:
```c
pthread_mutex_lock(&npu_mutex);
CVI_NN_Forward(model, inputs, n_in, outputs, n_out);
pthread_mutex_unlock(&npu_mutex);
```

### Rule 8: Calibration data format
`.npz` files used for INT8 calibration must use the exact input names matching ONNX:
```python
# Wake word model:
np.savez("sample_0000.npz", mel_input=X[i:i+1])   # key = input name

# GRU model (two inputs):
np.savez("sample_0000.npz",
         obs=X[i:i+1, np.newaxis, :],  # [1, 1, 16]
         hidden_in=zeros_hidden)        # [1, 64]
```

### Rule 9: cviruntime tensor name lookup
After INT8 quantization, tpu-mlir renames output tensors (e.g. `logits` → `logits_Gemm_f32`). Use wildcard matching:
```c
CVI_TENSOR *t = CVI_NN_GetTensorByName("logits*", outputs, n_out);
// Falls back to index 0 if name changed
if (!t) t = &outputs[0];
```

---

## 8. Phase 3 — Face Detection ✅ COMPLETE

**Goal:** Verify SCRFD face detection NPU timing and build the C decoder module.

**Model on board:** `/mnt/cvimodel/scrfd_768_432_int8_1x.cvimodel`  
**NPU latency: 12.2ms avg (81.7 FPS max)** ✅

### Key findings during Phase 3

The SDK binary `sample_vi_fd` (in `/root/`) **crashes** — it was compiled against an older model  
version and expects tensor names like `face_rpn_bbox_pred_stride32_dequant`, but the installed  
cvimodel uses newer names (`score_8_Sigmoid_dequant`, `bbox_8_Conv_dequant`, etc.).

**Solution:** We wrote our own `face_detector.c/h` with the correct tensor name decoding.

### SCRFD model tensor layout

Input:
- `input.1_quant_i8`  — [1, 3, 432, 768] INT8 BGR NCHW

Outputs (9 tensors):
```
score_8_Sigmoid_dequant   [54*96*2 = 10368]   ← confidence at stride-8 anchors
score_16_Sigmoid_dequant  [27*48*2 =  2592]
score_32_Sigmoid_dequant  [14*24*2 =   672]
bbox_8_Conv_dequant       [10368 * 4]           ← (dl,dt,dr,db) offsets
bbox_16_Conv_dequant      [ 2592 * 4]
bbox_32_Conv_dequant      [  672 * 4]
kps_8/16/32_Conv_dequant  (keypoints — unused)
```

### SCRFD BBox decode formula

```c
// For each anchor at (row, col) with stride s:
float cx = col * stride + stride * 0.5f;
float cy = row * stride + stride * 0.5f;
float x1 = cx - bbox[0] * stride;   // left
float y1 = cy - bbox[1] * stride;   // top
float x2 = cx + bbox[2] * stride;   // right
float y2 = cy + bbox[3] * stride;   // bottom
```

### Required LD_LIBRARY_PATH on board

```bash
export LD_LIBRARY_PATH=/mnt/system/usr/lib/3rd:/mnt/system/usr/lib:/mnt/system/lib:$LD_LIBRARY_PATH
```

(needs `libcvi_ive.so` from `/mnt/system/usr/lib/` and `libini.so` from `/mnt/system/usr/lib/3rd/`)

### Face detector C module

- `sg2002/face_detect/face_detector.h` — public API
- `sg2002/face_detect/face_detector.c` — SCRFD decoder + NMS
- `sg2002/face_detect/face_detect_test.c` — NPU verification test
- Binary on board: `/root/face_detect_test`

**Build:**
```bash
docker run --rm --network host \
  -v "/home/devinda_121260/Music/AI projects/pet_robot":/workspace \
  debian:9 bash /workspace/sg2002/face_detect/build.sh
```

**Run on board:**
```bash
export LD_LIBRARY_PATH=/mnt/system/usr/lib/3rd:/mnt/system/usr/lib:/mnt/system/lib:$LD_LIBRARY_PATH
./face_detect_test /mnt/cvimodel/scrfd_768_432_int8_1x.cvimodel
```

**Expected output:**
```
[FaceDetector] Ready: ...  (n_in=1 n_out=9)
Test 1 — blank frame:
  faces=0  person_visible=no  distance=0.000  latency=12694us
Latency benchmark (20 runs):
  min=12199us  max=12613us  avg=12244us  (81.7 FPS max)
PASS: avg 12244us < 50ms → running on NPU
```

### Integration plan (Phase 5)
- Run face detection at 10Hz in a separate pthread on SG2002
- Camera frame captured via Sophgo VI API → VPSS resize to 768×432 → INT8 normalize
- Write FaceResult to shared struct (guarded by mutex)
- Main GRU loop reads `person_visible` and `person_distance` from that struct

---

## 9. Phase 4 — ESP32-C3 Display Firmware ✅ COMPLETE

**Status:** Firmware written, built, and verified.  
**Project:** `esp32_display/` (separate from the earlier `esp32/` peripheral project).  
**Display:** ILI9341 TFT 320×240, **parallel 8080 interface** (NOT SPI).  
**Framework:** ESP-IDF v5.5.3 + FreeRTOS (NOT Arduino).

---

### 9.1 ESP-IDF Setup

Three versions are installed. **Always use v5.5.3:**

```bash
# First time only — install toolchain for ESP32-C3:
~/.espressif/v5.5.3/esp-idf/install.sh esp32c3

# Every session — source the environment:
source ~/.espressif/v5.5.3/esp-idf/export.sh
```

Installed versions for reference:

| Version | Path |
|---------|------|
| v6.1-dev (bleeding edge) | `~/esp/esp-idf/` |
| **v5.5.3 (use this)** | `~/.espressif/v5.5.3/esp-idf/` |
| v4.4.4 (too old) | `~/esp/v4.4.4/esp-idf/` |

---

### 9.2 Build and Flash

```bash
cd "/home/devinda_121260/Music/AI projects/pet_robot/esp32_display"

# Set target (only needed once per clean build dir):
idf.py set-target esp32c3

# Build:
idf.py build

# Find port:
ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null

# Flash and open serial monitor (exit with Ctrl+]):
idf.py -p /dev/ttyUSB0 flash monitor
```

**Expected boot output:**
```
I (267) nexu_display: Nexu Display C3 booting...
I (800) nexu_display: face animator running
I (800) nexu_display: ready.
```
Boot color test (red → green → blue flash) confirms display is alive before the face starts.

---

### 9.3 Wiring — ILI9341 Display → ESP32-C3 (Parallel 8080)

The display uses a **parallel 8-bit 8080 interface** — 8 data lines, not SPI.

| Signal | ESP32-C3 GPIO | Notes |
|--------|--------------|-------|
| D0 | GPIO0 | Data bus bit 0 |
| D1 | GPIO1 | Data bus bit 1 |
| D2 | GPIO2 | Data bus bit 2 |
| D3 | GPIO3 | Data bus bit 3 |
| D4 | GPIO4 | Data bus bit 4 |
| D5 | GPIO5 | Data bus bit 5 |
| D6 | GPIO6 | Data bus bit 6 |
| D7 | GPIO7 | Data bus bit 7 |
| WR | GPIO8 | Write strobe (active low) |
| CS | GPIO9 | Chip select (active low) |
| DC | GPIO10 | 0=command, 1=data |
| RST | 3.3V | Tied high — SW reset via cmd 0x01 |
| RD | 3.3V | Tied high — read not used |
| BL | 3.3V | Backlight hardwired on |

Inter-C3 UART (Main C3 → Display C3):

| Signal | GPIO |
|--------|------|
| UART1 RX | GPIO20 |
| UART1 TX | GPIO21 (optional debug) |

> GPIO18/19 are USB D-/D+ on ESP32-C3 — **do not use**.

---

### 9.4 Wiring — TTP223 Touch Sensor → ESP32-C3

| TTP223 Pin | ESP32-C3 |
|------------|----------|
| VCC | 3.3V |
| GND | GND |
| SIG (OUT) | GPIO2 |

> GPIO2 is configured with internal pull-down. When nothing is connected, the pin floats and logs spurious "Head touch!" events every 500ms (the cooldown period). This is expected — it stops as soon as the TTP223 is wired and driving a clean digital signal.

**Temporarily disable touch during display-only testing** (in [main.c](esp32/main/main.c)):
```c
// touch_init();   /* re-enable when TTP223 is wired */
```

---

### 9.5 Firmware Architecture (`esp32_display/`)

Two tasks — this C3 is dedicated entirely to the display:

| Task | Priority | Stack | Role |
|------|----------|-------|------|
| display_task | 4 | 4096 | boot color test → face_init → face_tick every 33ms (30fps) |
| uart_cmd (task inside uart_cmd_init) | 5 | 2048 | reads UART1 GPIO20, calls face_set() on DISPLAY: lines |

Data flow:
```
Main C3  →  UART1 RX (GPIO20)  →  uart_cmd task  →  face_set()  →  display_task → ILI9341
```

**Source files:**

| File | Purpose |
|------|---------|
| `ili9341.h/c` | Parallel 8080 driver: `set_window`, `write_color`, `write_pixels`, `stream_pixels`/`stream_end` |
| `gfx.h/c` | `fill_rect`, `fill_circle`, `fill_ellipse`, `draw_arc`, `draw_line`, `fill_rounded_rect` |
| `face.h/c` | 5-expression animated face with full scanline compositor |
| `uart_cmd.h/c` | UART1 receiver: parses `DISPLAY:expr\n`, calls `face_set()` |
| `main.c` | Boots display, starts uart_cmd, runs display_task at 30fps |

---

### 9.6 UART Line Protocol

**SG2002 → ESP32-C3** (commands):
```
DISPLAY:happy\n      → set face expression (smooth transition)
DISPLAY:curious\n
DISPLAY:alert\n
DISPLAY:sleep\n
DISPLAY:idle\n
PLAY:vocal_happy.wav\n   → synthesize vocal sound on ESP32 (PCM5102A)
PLAY:vocal_curious.wav\n
PLAY:vocal_alert.wav\n
PLAY:vocal_play.wav\n
MOVE:fwd,60\n            → motor (future)
```

**ESP32-C3 → SG2002** (events):
```
TOUCH:head\n         → head touch triggered (500ms cooldown)
TOUCH:body\n         → (future)
COLLISION:front\n    → (future)
ENCODER:100,98\n     → (future)
```

Settings: UART1, GPIO20=RX, GPIO21=TX, 921600 baud, 8N1.

---

### 9.7 Animated Face System

The face is **never static**. Every frame something moves. This is what makes Nexu feel alive.

**5 expressions:**

| Name | Eyes | Mouth | Cheeks | Brows |
|------|------|-------|--------|-------|
| idle | Round (rx=32, ry=32) | Gentle arc (r=30, 20°–100°) | No | No |
| happy | Slightly squinted (ry=30) | Wide smile (r=38, 30°–150°) | Yes (pink) | No |
| curious | Left normal + brow; right larger (ry=38) + angled brow 12° | Small arc | No | Yes |
| alert | Wide ellipse (rx=38, ry=40), pupil ×1.3, brows 8° | O-ring (r=16) | No | Yes |
| sleep | Squished slits (ry=5), Zzz bubbles | Tiny arc | No | No |

**Animation layers running simultaneously every frame:**

1. **Expression morph** — lerp at t=0.08 per frame (~500ms full transition at 30fps)
2. **Blink** — random interval 2–6 seconds, 250ms duration, smooth close/open
3. **Pupil drift** — each eye drifts independently ±5px via sine waves at different phases
4. **Breathing** — eye vertical radius pulses ±5% at ~0.18Hz
5. **Scanline compositor** — one `set_window` per region per frame, streaming pixels with CS held low

**Rendering architecture (flicker-free):**
- Render order: **mouth first, then left eye, then right eye**
  - Eyes render last so they always overwrite the mouth's BG fill in the overlap band
- Each region uses one `ili9341_set_window(bx, by, bx_end, by_end)` then streams all rows with `ili9341_stream_pixels` (CS stays LOW), then `ili9341_stream_end()` once
- ~10× fewer bus transactions vs per-row `set_window`
- Mouth uses a fixed oversized BB (±55px around MOUTH_BASE_CY) — no prev-geometry tracking needed

**Face API:**
```c
face_init();                  // call once — clears screen, sets IDLE state
face_set(FACE_HAPPY);         // set target — smooth transition, not instant
face_tick();                  // call every 33ms — advances all animations
face_from_string("happy");    // parses UART arg → FaceExpression
```

**Layout (landscape 320×240):**
```
Left eye center:   (105, 105)
Right eye center:  (215, 105)
Mouth center:      (160, 178)
Brow:              cy - ry - 18 ± 10*sin(angle), cx±20
Cheeks (HAPPY):    cy + 45, 22×10 ellipse
Zzz (SLEEP):       top-right corner, 3 pulsing circles
```

---

### 9.8 Known Issues and Fixes (ILI9341 parallel version)

**Brow discontinuities on angled expressions (ALERT, CURIOUS):**
- Root cause: old rasterizer drew one 5px segment per Y row; a 40px-wide 4-row-tall angled brow left 5–10px gaps between segments
- Fix: t_lo/t_hi sweep — for each Y row, compute the full X span the thick line covers, fill it entirely

**Black boxes appearing in mouth area during expression transitions:**
- Root cause: mouth rendered after eyes; mouth's BG fill clobbered the eye/cheek overlap band (~y 133–137)
- Fix: render mouth first, eyes last

**Mouth bounding-box artifacts when switching between arc and O-ring:**
- Root cause: `prev_r` tracked last frame's radius; when `is_circle` flipped, `cur_r` reset to ~5 and the old arc wasn't erased
- Fix: fixed oversized BB (±55px) for mouth — always covers any previous geometry

**Streaming API for smooth refresh:**
- Old: `ili9341_set_window` per row = 11 bytes of address overhead per row (col addr + row addr + write cmd)
- New: `ili9341_stream_pixels` / `ili9341_stream_end` — one set_window per region, CS held low across all rows
- Result: ~10× reduction in bus overhead per frame

---

### 9.9 Audio — PCM5102A I2S DAC (ESP32-C3)

**Component chain:** ESP32-C3 I2S → PCM5102A (DAC) → PAM8403 (amp) → 8Ω speaker  
**Source file:** `esp32/main/audio/i2s_audio.h/c`  
**No WAV files needed** — sounds are synthesized in real-time on the ESP32.

#### PCM5102A Wiring

| PCM5102A pin | ESP32-C3 GPIO | Notes |
|---|---|---|
| BCLK (BCK) | GPIO0 | I2S bit clock |
| LRCK | GPIO1 | I2S word select |
| DIN | GPIO10 | I2S serial data |
| SCK | GND | Auto-clock mode (no MCLK needed) |
| VCC | 3.3V | |
| GND | GND | |

#### PAM8403 Wiring

| PAM8403 pin | Connected to |
|---|---|
| VCC | 5V |
| GND | GND |
| L+/L− IN | PCM5102A LOUT+/LOUT− |
| Speaker+ | Speaker terminal |
| Speaker− | Speaker terminal |

#### Synthesized Vocal Sounds

| PLAY command | Sound character | Synthesis |
|---|---|---|
| `PLAY:vocal_happy.wav` | Cheerful arpeggio | C4→E4→G4→C5 notes (80–140ms each) |
| `PLAY:vocal_curious.wav` | Rising glide | 280→480 Hz sweep over 300ms |
| `PLAY:vocal_alert.wav` | Warning beeps | Two 880 Hz pulses × 100ms |
| `PLAY:vocal_play.wav` | Playful wobble | 380↔580 Hz up-down-up |

The `.wav` suffix is stripped by the ESP32 audio task before matching — so `vocal_happy.wav` and `vocal_happy` both work.

#### Audio task integration

```
SG2002 UART → uart_rx_task → s_cmd_queue → display_task → s_audio_queue → audio_task → I2S DMA → PCM5102A
```

`audio_task` runs at priority 3 (below display at 4) so it never interferes with animation.  
Sample rate: 44100 Hz, 16-bit stereo. Synthesis uses `sinf()` + linear frequency sweep.

---

## 10. Phase 5 — SG2002 Main Loop ✅ COMPLETE

**Status:** Written, cross-compiled, and tested on board. Binary: 21KB.

**Source:** `sg2002/main_loop/` — 6 C modules + Makefile + build.sh.

### 10.1 Files

| File | Purpose |
|---|---|
| `obs.h` | `ObsVector` struct (16 floats), `obs_to_array()` helper |
| `inference.h/c` | Loads both models at startup, NPU mutex, `wake_word_run()`, `gru_run()` |
| `uart_sg.h/c` | `/dev/ttyS1` at 921600 baud, RX background thread, event flags, `uart_sg_send()` |
| `audio.h/c` | `arecord` ring buffer, radix-2 FFT, mel filterbank, `audio_get_mel()` |
| `action.h/c` | Temperature sampling (T=0.8), 12-action → UART command dispatch |
| `main.c` | 10Hz main loop, personality init, Ctrl+C cleanup |

### 10.2 Main Loop

Runs at 10Hz (100ms sleep). Wake word check every 10 ticks (every 1 second).

```
boot:
  load wake_word.cvimodel + gru_behavior.cvimodel onto NPU
  open /dev/ttyS1 @ 921600 baud
  start audio capture thread (arecord hw:1,0)
  randomize personality: curiosity ∈ [0.4, 0.9], playful ∈ [0.4, 0.9]

every 100ms:
  consume UART events (TOUCH:head, COLLISION:front, ...)
  every 1s: audio_get_mel() → wake_word_run() → set obs.wake_word
  build ObsVector from all sensor fields
  gru_run(obs, hidden) → action_logits, new_hidden
  action_sample(logits, T=0.8)  ← temperature sampling, not argmax
  action_dispatch(action)        ← sends DISPLAY:/PLAY: over UART

Ctrl+C: send DISPLAY:sleep\n before exit
```

### 10.3 Audio Capture — `audio.c`

No libasound dependency. Captures via `popen("arecord -D hw:1,0 -r 16000 -c 1 -f S16_LE")`.

```
SAMPLE_RATE = 16000 Hz
N_FFT       = 512   (32ms window)
HOP_LENGTH  = 160   (10ms hop)
N_MELS      = 40
N_FRAMES    = 98    (~1 second)
NEEDED_SAMPLES = N_FFT + (N_FRAMES-1)*HOP = 16192
RING_SIZE      = 17000
```

Capture thread reads 160 samples (one hop) at a time into a ring buffer. `audio_get_mel()` copies the most recent 16192 samples and runs:
1. Hann window: `s_hann[i] = 0.5*(1 − cos(2πi/(N_FFT−1)))`
2. Radix-2 Cooley-Tukey FFT (in-place, N=512)
3. Power spectrum → mel filterbank (40 triangular filters, librosa-compatible)
4. Log + normalize: zero mean, unit variance

Returns `false` (zeroed mel output) if mic is not connected — loop continues safely without audio input.

### 10.4 Inference — `inference.c`

```c
// NPU mutex — CV181x single-context; only one Forward at a time
pthread_mutex_t g_npu_mutex;

// Wake word: returns true if logits[1] (wake) > logits[0] (no_wake)
bool wake_word_run(const float *mel);   // input: MEL_ELEMS = 40*98

// GRU: updates hidden in-place, writes 12 action logits
int gru_run(const float *obs_arr, float *hidden, float *logits);
```

Both models run on the NPU (sub-millisecond). Models must be on board at:
- `/root/wake_word.cvimodel`
- `/root/gru_behavior.cvimodel`

### 10.5 Build and Deploy

**Cross-compile:**
```bash
docker run --rm --network host \
  -v "/home/devinda_121260/Music/AI projects/pet_robot":/workspace \
  debian:9 bash /workspace/sg2002/main_loop/build.sh
```

**Transfer to board:**
```bash
cd "/home/devinda_121260/Music/AI projects/pet_robot/sg2002/main_loop"
python3 -m http.server 8769 --bind 0.0.0.0 &
sshpass -p milkv ssh root@192.168.8.118 \
  "wget -q http://192.168.8.124:8769/nexu_main -O /root/nexu_main && chmod +x /root/nexu_main"
kill %1
```

**Run on board:**
```bash
sshpass -p milkv ssh root@192.168.8.118 \
  "LD_LIBRARY_PATH=/mnt/system/usr/lib/3rd:/mnt/system/usr/lib:/mnt/system/lib \
   ./nexu_main /root/wake_word.cvimodel /root/gru_behavior.cvimodel"
```

**Confirmed output from board:**
```
[inference] Wake word model ready
[inference] GRU model ready
[uart_sg] Opened /dev/ttyS1 @ 921600 baud
[main] Personality: curiosity=0.60  playful=0.77
[action] play_gesture
[action] idle_sit
[action] look_around
[action] idle_sit
...
```

### 10.6 What's waiting for hardware

| Feature | Needs | Status |
|---|---|---|
| Wake word detection | USB mic (`hw:1,0`) | Arrives ~2 days |
| Speaker audio | PCM5102A + PAM8403 + speaker | Arrives ~2 days |
| Animated face display | ST7789V wiring soldered | FPC adapter on order |
| Touch events | TTP223 wired to ESP32-C3 GPIO2 | Needs soldering |
| SG2002↔ESP32 UART | Cable wired between boards | Needs soldering |

When all hardware arrives: wire everything, `idf.py flash` the ESP32, run `nexu_main` on the SG2002, and Nexu is alive.

---

## 11. Phase 6 — Integration (Pending)

**Integration testing checklist:**
- [ ] Wake word → ESP32 display shows "!" expression
- [ ] Touch head → Nexu vocalizes happy
- [ ] Person visible → Nexu tracks (look_around / approach)
- [ ] Long idle → Nexu self-grooms or sleeps
- [ ] Two models running simultaneously (wake word + GRU) without NPU conflict
- [ ] Temperature sampling produces varied, natural behavior
- [ ] Personality persists across reboots (save to file)

**Tuning levers:**
- `temperature` in action sampling: lower = more deterministic, higher = more random
- `personality_curiosity` and `personality_playful`: set at boot, optionally randomized
- `idle_ticks` threshold: controls how quickly Nexu gets restless

---

## Quick Reference — All Commands

### Training pipeline
```bash
cd "/home/devinda_121260/Music/AI projects/pet_robot"

# Wake word
conda run -n pet_robot python wake_word_model/model/data_pipeline.py
conda run -n pet_robot python wake_word_model/model/train.py
conda run -n pet_robot python wake_word_model/export/export_onnx.py

# GRU behavior
conda run -n pet_robot python gru_behavior_model/model/simulate.py
conda run -n pet_robot python gru_behavior_model/model/train.py
conda run -n pet_robot python gru_behavior_model/export/export_onnx.py
```

### tpu-mlir conversion
```bash
# Wake word
docker run --rm --network host \
  -v "/home/devinda_121260/Music/AI projects/pet_robot/wake_word_model":/workspace \
  -v "/tmp/tpu_wheels":/wheels \
  sophgo/tpuc_dev:latest bash /workspace/convert/convert.sh

# GRU behavior
docker run --rm --network host \
  -v "/home/devinda_121260/Music/AI projects/pet_robot/gru_behavior_model":/workspace \
  -v "/tmp/tpu_wheels":/wheels \
  sophgo/tpuc_dev:latest bash /workspace/convert/convert.sh
```

### Cross-compile binaries
```bash
# NPU test binaries
docker run --rm --network host \
  -v "/home/devinda_121260/Music/AI projects/pet_robot":/workspace \
  debian:9 bash /workspace/sg2002/wake_word/build.sh

docker run --rm --network host \
  -v "/home/devinda_121260/Music/AI projects/pet_robot":/workspace \
  debian:9 bash /workspace/sg2002/gru_behavior/build.sh

# Phase 5 — SG2002 main loop (nexu_main)
docker run --rm --network host \
  -v "/home/devinda_121260/Music/AI projects/pet_robot":/workspace \
  debian:9 bash /workspace/sg2002/main_loop/build.sh
```

### Transfer files to board
```bash
# Serve files from directory:
python3 -m http.server 8765 --bind 0.0.0.0 &

# On board fetch:
sshpass -p milkv ssh root@192.168.8.118 \
  "wget -q http://192.168.8.124:8765/FILENAME -O /root/FILENAME && chmod +x /root/FILENAME"

kill %1
```

### Verify on board
```bash
sshpass -p milkv ssh root@192.168.8.118 "./wake_word_test wake_word.cvimodel"
sshpass -p milkv ssh root@192.168.8.118 "./gru_behavior_test gru_behavior.cvimodel"

# Run the full main loop:
sshpass -p milkv ssh root@192.168.8.118 \
  "LD_LIBRARY_PATH=/mnt/system/usr/lib/3rd:/mnt/system/usr/lib:/mnt/system/lib \
   ./nexu_main /root/wake_word.cvimodel /root/gru_behavior.cvimodel"
```

### ESP32-C3 display firmware build and flash
```bash
source ~/.espressif/v5.5.3/esp-idf/export.sh
cd "/home/devinda_121260/Music/AI projects/pet_robot/esp32_display"
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor
```

---

*Last updated: Phase 5 complete — SG2002 main loop written, compiled (21KB), and tested on board at 10Hz.*  
*Phase 4 updated: Display changed to ILI9341 parallel 8080. Face animator rewritten with streaming API (no per-row set_window), brow rasterizer fixed (t_lo/t_hi sweep), mouth black-box bug fixed (render order), and mouth transition artifacts eliminated (fixed oversized BB).*  
*Waiting for hardware: USB mic + PCM5102A + PAM8403 + speaker, and ILI9341 display wiring.*  
*Next: Phase 6 — wire everything up and run full integration.*
