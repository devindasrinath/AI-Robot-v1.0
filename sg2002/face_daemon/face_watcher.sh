#!/bin/sh
#
# face_watcher.sh — runs sample_vi_fd and writes face detection results
# to /tmp/nexu_face for nexu_main to read.
#
# Usage (run before nexu_main):
#   chmod +x face_watcher.sh
#   ./face_watcher.sh &
#
# Output file format: "<visible> <distance>\n"
#   visible  : 1 if face(s) detected, 0 if not
#   distance : 0.50 when visible (fixed for now — no bbox from stdout)
#              0.00 when no face
#
# sample_vi_fd only prints "face count: N" when the count changes,
# so the file correctly reflects the latest state between changes.

RESULT_FILE=/tmp/nexu_face
MODEL=/mnt/cvimodel/scrfd_768_432_int8_1x.cvimodel
BINARY=/root/sample_vi_fd

export LD_LIBRARY_PATH=/mnt/system/usr/lib/3rd:/mnt/system/usr/lib:/mnt/system/lib:$LD_LIBRARY_PATH

if [ ! -f "$BINARY" ]; then
    echo "[face_watcher] ERROR: $BINARY not found" >&2
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "[face_watcher] ERROR: $MODEL not found" >&2
    exit 1
fi

# Start with no face
printf "0 0.00\n" > "$RESULT_FILE"
echo "[face_watcher] Started. Writing to $RESULT_FILE"

"$BINARY" "$MODEL" 2>/dev/null | while IFS= read -r line; do
    case "$line" in
        "face count: "*)
            count="${line#face count: }"
            if [ "$count" -gt 0 ] 2>/dev/null; then
                printf "1 0.50\n" > "$RESULT_FILE"
            else
                printf "0 0.00\n" > "$RESULT_FILE"
            fi
            ;;
    esac
done

echo "[face_watcher] sample_vi_fd exited"
printf "0 0.00\n" > "$RESULT_FILE"
