# Nexu Pet Robot — Project Context for Claude Code

## Auto-Update Rule
After completing any significant task, automatically run:
`echo "- $(date): [what was done]" >> CLAUDE.md`
Or update the Session Log section in this file directly.

## Session Log
<!-- Auto-updated by Claude -->

Add this section to CLAUDE.md:

## Auto-Documentation Rule
After every successfully completed task or phase, append a summary to GUIDE.md using this exact format:

### [Task Name] — [Date]
**What we did:** [1-2 sentences]
**Commands used:** [only new/changed commands]
**Key decisions:** [any important choices made]
**Status:** ✅ Complete / 🔄 In Progress

Do this automatically without being asked. Only append — never overwrite existing content.
Also update the "Last updated" line at the bottom of GUIDE.md after each session.

### Session — 2026-05-01: ILI9341 display firmware + face animation polish

**What was built:**
- `esp32_display/` — new ESP-IDF project for a second ESP32-C3 dedicated to the display
  - `ili9341.h/c` — ILI9341 parallel 8080 driver (D0-D7 = GPIO 0-7, WR=8, CS=9, DC=10)
  - `gfx.h/c` — primitive drawing (fill_rect, fill_circle, fill_ellipse, draw_arc, draw_line)
  - `face.h/c` — fully animated face: 5 expressions, lerp morph, blink, pupil drift, breathing
  - `uart_cmd.h/c` — UART1 receiver (GPIO20 RX) parses DISPLAY:expr lines from Main C3
  - `main.c` — display task at 30fps + self-test cycle at boot

**Face rendering bugs fixed this session:**
1. **Black boxes in mouth area** — root cause: mouth rendered after eyes; its BG fill clobbered the eye/cheek overlap band. Fix: render mouth first, eyes last.
2. **Brow discontinuities** — root cause: parametric rasterizer drew one 5px segment per Y row, leaving 5–10px horizontal gaps on angled brows. Fix: t_lo/t_hi span sweep fills the full X extent at each Y.
3. **Flickering + slow refresh** — root cause: `ili9341_set_window` called once per row (11-byte overhead each). Fix: `ili9341_stream_pixels` / `ili9341_stream_end` streaming API — one set_window per region, CS held low across all rows (~10× less bus overhead).
4. **Mouth transition artifacts** — root cause: prev_r lost track of arc radius when `is_circle` mode flipped. Fix: mouth uses a fixed oversized BB (±55px around MOUTH_BASE_CY) — no prev geometry tracking needed.

**Hardware note:** Display is ILI9341 (parallel 8080 interface), NOT the originally planned ST7789V SPI.

**Build verified:** `idf.py build` passes clean (only pre-existing unused-function warning).

## What this project is
A pet robot named "Nexu" — a unique cute personality (NOT a cat or dog). Runs fully on-device:
- **SG2002** (Milk-V Duo S) — main brain, runs all ML inference on CV181x NPU
- **ESP32-C3** — peripheral controller (display, speaker, touch sensor, motors)
- No cloud. No LLM. Everything local.

## Board access
- IP: `192.168.8.119`, password: `milkv`  *(changes with network — check arp/nmap if unreachable)*
- SSH: `sshpass -p milkv ssh root@192.168.8.160 "command"`
- File transfer (scp broken — sftp-server missing): use HTTP server + wget
  ```bash
  python3 -m http.server 8765 --bind 0.0.0.0 &
  sshpass -p milkv ssh root@192.168.8.118 "wget -q http://192.168.8.124:8765/file -O /root/file"
  ```

## Hardware confirmed by user
| Component | Details |
|-----------|---------|
| Camera | GC2083 (officially supported by Milk-V SDK) |
| Display | 2.4" TFT SPI 240×320 ST7789V (10-pin FPC, FPC adapter board on order from AliExpress) → ESP32-C3 |
| Mic | USB mic ordered (local dealer, ~2 days) — ALSA `hw:1,0` |
| Speaker | PCM5102A I2S DAC + PAM8403 Class-D amp + 8Ω speaker → ESP32-C3 (ordered, ~2 days) |
| Touch sensor | Capacitive TTP223 on Nexu's head → ESP32-C3 GPIO2 → UART |
| Motors | 2 encoder motors → ESP32-C3 (skipped for initial phase, robot stationary) |
| Obstacle sensors | Not decided yet (HC-SR04 or VL53L0X) |
| UART | 921600 baud between SG2002 ↔ ESP32-C3 |

## PCM5102A wiring (ESP32-C3)
| PCM5102A pin | ESP32-C3 GPIO |
|---|---|
| BCLK (BCK) | GPIO0 |
| LRCK | GPIO1 |
| DIN | GPIO10 |
| SCK | GND (auto-clock mode — no MCLK needed) |
| VCC | 3.3V |
| GND | GND |

## ML Models — all trained + NPU verified

### Model A: Wake Word DS-CNN
- **Purpose:** Detect "Nexu" from microphone audio
- **Input:** `mel_input` [1, 1, 40, 98] — log-mel spectrogram (1s @ 16kHz, 40 mel bins)
- **Output:** `logits` [1, 2] — [no_wake, wake]
- **File:** `wake_word_model/wake_word.cvimodel` (25KB)
- **Test binary on board:** `/root/wake_word_test`
- **NPU latency:** 182µs avg ✅
- **Training:** recall=1.000, fpr=0.000 (294 pos TTS + 1600 neg GSC samples)
- **Architecture:** DS-CNN with `AvgPool2d(kernel_size=(5,13))` — NOT AdaptiveAvgPool2d

### Model B: GRU Behavior Policy
- **Purpose:** Decide what Nexu does next based on sensors + internal state
- **Input:** `obs` [1, 1, 16], `hidden_in` [1, 64]
- **Output:** `action_logits` [1, 12], `hidden_out` [1, 64]
- **File:** `gru_behavior_model/gru_behavior.cvimodel` (39KB)
- **Test binary on board:** `/root/gru_behavior_test`
- **NPU latency:** 125µs avg ✅
- **Architecture:** ManualGRUCell (explicit linear ops — NOT nn.GRU which tpu-mlir rejects)

#### Observation vector [16 floats]:
```
0  touch_head        1  touch_body        2  touch_tail
3  collision_front   4  collision_left    5  collision_right
6  person_visible    7  person_distance   8  wake_word
9  battery_level    10  time_since_touch  11  idle_ticks_norm
12 sound_level      13 motion_energy     14 personality_curiosity
15 personality_playful
```

#### Actions [12]:
```
0 idle_sit    1 look_around   2 approach      3 follow
4 play_gesture 5 vocal_happy  6 vocal_curious  7 vocal_alert
8 sleep        9 avoid_obstacle 10 self_groom  11 wag_tail
```

### Model C: SCRFD Face Detection ✅ NPU verified (12.2ms)
- cvimodel on board: `/mnt/cvimodel/scrfd_768_432_int8_1x.cvimodel`
- Custom C module: `sg2002/face_detect/face_detector.c/h` (SCRFD decoder + NMS)
- Test binary on board: `/root/face_detect_test`
- Requires: `LD_LIBRARY_PATH=/mnt/system/usr/lib/3rd:/mnt/system/usr/lib:/mnt/system/lib`
- WARNING: SDK `sample_vi_fd` crashes — tensor name mismatch (old binary, new model). Ignore it.
- Camera capture via Sophgo VI API done in Phase 5

## Critical tpu-mlir rules (learned the hard way)
1. **AdaptiveAvgPool2d → BANNED** — exports as ReduceMean which tpu-mlir rejects. Use `nn.AvgPool2d(kernel_size=(H,W))` with exact feature map dims.
2. **nn.GRU/nn.LSTM → BANNED** — use ManualGRUCell (explicit sigmoid/tanh/matmul).
3. **ONNX opset 13** — must use `dynamo=False` in torch.onnx.export (new exporter can't target opset 13).
4. **Static shapes** — no dynamic axes anywhere. tpu-mlir can't handle dynamic shapes.
5. **GenericCpu `quant` ops are OK** — just the INT8 boundary quantization, not a CPU fallback.
6. **Docker for conversion** — `sophgo/tpuc_dev:latest` with `--network host` for DNS.
7. **Local wheel** — `/tmp/tpu_wheels/tpu_mlir-1.27-py3-none-any.whl` (mount as `/wheels`).
8. **NPU single-context** — serialize all `CVI_NN_Forward` calls with mutex in production.

## Environments
- **Training:** conda env `pet_robot` (Python 3.12)
- **Conversion:** Docker `sophgo/tpuc_dev:latest` (Python 3.10, tpu-mlir pre-installed)
- **Cross-compile:** `docker run debian:9` + `gcc-aarch64-linux-gnu`, then HTTP-transfer binary to board

## Conversion command
```bash
docker run --rm --network host \
  -v "/home/devinda_121260/Music/AI projects/pet_robot/wake_word_model":/workspace \
  -v "/tmp/tpu_wheels":/wheels \
  sophgo/tpuc_dev:latest \
  bash /workspace/convert/convert.sh
```

## Cross-compile command
```bash
docker run --rm --network host \
  -v "/home/devinda_121260/Music/AI projects/pet_robot":/workspace \
  debian:9 bash /workspace/sg2002/<module>/build.sh
```

## Project directory structure
```
pet_robot/
├── wake_word_model/
│   ├── model/         wake_word_cnn.py, data_pipeline.py, train.py
│   ├── export/        export_onnx.py, wake_word.onnx
│   ├── convert/       convert.sh
│   ├── data/          X_train.npy, y_train.npy, X_val.npy, y_val.npy
│   ├── calibration_data/samples/  (100 .npz files)
│   └── wake_word.cvimodel  ← deployed to board
├── gru_behavior_model/
│   ├── model/         gru_policy.py, simulate.py, train.py
│   ├── export/        export_onnx.py, gru_behavior.onnx
│   ├── convert/       convert.sh
│   ├── data/          X_train.npy, y_train.npy, X_val.npy, y_val.npy
│   ├── calibration_data/samples/  (200 .npz files)
│   └── gru_behavior.cvimodel  ← deployed to board
├── sg2002/
│   ├── include/       cviruntime.h, cvitpu_debug.h
│   ├── lib/           libcviruntime.so, libcvikernel.so
│   ├── wake_word/     wake_word_test.c, Makefile, build.sh
│   ├── gru_behavior/  gru_behavior_test.c, Makefile, build.sh
│   ├── face_detect/   face_detector.h/c, face_detect_test.c, Makefile, build.sh
│   └── main_loop/     main.c, inference.h/c, uart_sg.h/c, audio.h/c, action.h/c, obs.h, Makefile, build.sh
└── esp32/
    ├── CMakeLists.txt
    ├── sdkconfig.defaults
    └── main/
        ├── main.c
        ├── display/   st7789.h/c, gfx.h/c, face.h/c
        ├── uart_comm/ uart_comm.h/c
        ├── sensors/   touch.h/c
        └── audio/     i2s_audio.h/c  ← I2S + PCM5102A + tone synthesis
```

## Phase status
- ✅ Phase 0: Infrastructure (Docker, conda env, board SSH, cviruntime libs)
- ✅ Phase 1: Wake word DS-CNN — trained, converted, NPU verified (182µs)
- ✅ Phase 2: GRU behavior policy — trained, converted, NPU verified (125µs)
- ✅ Phase 3: Face detection — SCRFD NPU verified (12.2ms), custom C decoder written
- ✅ Phase 4: ESP32-C3 firmware — display + UART + touch + I2S audio — built, boot-verified (display wiring pending)
- ✅ Phase 5: SG2002 main loop — written, cross-compiled (21KB), tested on board at 10Hz ✅
- 🔲 Phase 6: Integration + tuning (waiting for display + mic + speaker hardware)
