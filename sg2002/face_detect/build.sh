#!/usr/bin/env bash
set -euo pipefail

cat > /etc/apt/sources.list << 'SOURCES'
deb http://archive.debian.org/debian stretch main
deb http://archive.debian.org/debian-security stretch/updates main
SOURCES

apt-get -o Acquire::Check-Valid-Until=false update -qq 2>&1 | tail -3
apt-get install -y -qq gcc-aarch64-linux-gnu make 2>&1 | tail -3

cd /workspace/sg2002/face_detect
make clean 2>/dev/null || true
make
echo "Binary size: $(stat -c%s face_detect_test) bytes"
