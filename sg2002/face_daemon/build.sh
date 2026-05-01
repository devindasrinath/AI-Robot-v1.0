#!/usr/bin/env bash
# Build sample_vi_nexu_face for aarch64 (Milk-V Duo S / SG2002)
# Run inside debian:9 Docker:
#   docker run --rm -v "/path/to/pet_robot":/workspace debian:9 bash /workspace/sg2002/face_daemon/build.sh
set -euo pipefail

cat > /etc/apt/sources.list << 'SOURCES'
deb http://archive.debian.org/debian stretch main
deb http://archive.debian.org/debian-security stretch/updates main
SOURCES

apt-get -o Acquire::Check-Valid-Until=false update -qq 2>&1 | tail -3
apt-get install -y -qq gcc-aarch64-linux-gnu make 2>&1 | tail -3

WORKSPACE=/workspace
SDK_ROOT=$WORKSPACE/cvitek-tdl-sdk-sg200x
MW_INC=$SDK_ROOT/sample/3rd/middleware/v2/include
UTILS=$SDK_ROOT/sample/utils
TDL_INC=$SDK_ROOT/include/cvi_tdl
SDK_INC=$SDK_ROOT/include
BOARD_LIBS=$WORKSPACE/sg2002/board_libs
FACE_DET=$WORKSPACE/sg2002/face_detect
SRC=$SDK_ROOT/sample/cvi_tdl/sample_vi_nexu_face.c
OUT=$WORKSPACE/sg2002/face_daemon/sample_vi_nexu_face

CC=aarch64-linux-gnu-gcc
CHIP_DEF="-DCV181X -D_MIDDLEWARE_V2_"

CFLAGS="$CHIP_DEF -O2 -std=gnu11 -fsigned-char -Wno-pointer-to-int-cast \
  -I$MW_INC \
  -I$MW_INC/isp/cv181x \
  -I$TDL_INC \
  -I$SDK_INC \
  -I$UTILS \
  -I$FACE_DET \
  -I$WORKSPACE/sg2002/include \
  -I$SDK_ROOT/sample/3rd/rtsp/include \
  -I$SDK_ROOT/sample/3rd/rtsp/include/cvi_rtsp"

# Compile utility objects (skip sample_utils.c — only macros needed, YOLO calls break link)
$CC $CFLAGS -c $UTILS/middleware_utils.c    -o /tmp/middleware_utils.o
$CC $CFLAGS -c $UTILS/vi_vo_utils.c         -o /tmp/vi_vo_utils.o
$CC $CFLAGS -c $FACE_DET/face_detector.c    -o /tmp/face_detector.o

# Compile and link
$CC $CFLAGS \
  $SRC \
  /tmp/middleware_utils.o \
  /tmp/vi_vo_utils.o \
  /tmp/face_detector.o \
  -L$BOARD_LIBS \
  -lcvi_tdl -lcviruntime -lcvikernel -lcvimath \
  -lsys -lvpss -lsample -lisp -lsns_full \
  -lmisc -lisp_algo -lvdec -lvenc \
  -lae -laf -lawb -lcvi_bin -lcvi_bin_isp \
  -lvi -lvo -lrgn -lgdc -lcvi_ive -lcvi_rtsp \
  -lopencv_core -lopencv_imgproc -lopencv_imgcodecs \
  -lini -lpthread -lm -lz -latomic \
  -Wl,-rpath,/mnt/system/lib:/mnt/system/usr/lib:/mnt/system/usr/lib/3rd \
  -Wl,--allow-shlib-undefined \
  -o $OUT

echo "Built: $OUT ($(stat -c%s $OUT) bytes)"
