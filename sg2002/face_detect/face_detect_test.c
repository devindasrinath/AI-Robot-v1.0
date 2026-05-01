/*
 * face_detect_test.c — Verify SCRFD face detection on SG2002 NPU
 *
 * Build: make -C sg2002/face_detect/
 * Run:   ./face_detect_test /mnt/cvimodel/scrfd_768_432_int8_1x.cvimodel
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "face_detector.h"

static long now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000L + ts.tv_nsec / 1000L;
}

int main(int argc, char *argv[])
{
    const char *model = (argc > 1) ? argv[1]
        : "/mnt/cvimodel/scrfd_768_432_int8_1x.cvimodel";

    printf("=== SCRFD Face Detection NPU Verification ===\n");
    printf("Model: %s\n\n", model);

    FaceDetector *fd = face_detector_create(model);
    if (!fd) {
        fprintf(stderr, "Failed to create detector\n");
        return 1;
    }

    /* Test 1: blank frame (no faces expected) */
    FaceResult result;
    long t0 = now_us();
    face_detector_run(fd, NULL, &result);
    long lat = now_us() - t0;
    printf("Test 1 — blank frame:\n");
    printf("  faces=%d  person_visible=%s  distance=%.3f  latency=%ldus\n\n",
           result.num_faces,
           result.person_visible ? "YES" : "no",
           result.person_distance, lat);

    /* Test 2: latency benchmark (20 runs) */
    printf("Latency benchmark (20 runs):\n");
    long total = 0, mn = 1000000L, mx = 0;
    for (int i = 0; i < 20; i++) {
        t0 = now_us();
        face_detector_run(fd, NULL, &result);
        long e = now_us() - t0;
        total += e;
        if (e < mn) mn = e;
        if (e > mx) mx = e;
    }
    long avg = total / 20;
    printf("  min=%ldus  max=%ldus  avg=%ldus  (%.1f FPS max)\n\n",
           mn, mx, avg, 1e6f / avg);

    if (avg < 50000)
        printf("PASS: avg %ldus < 50ms → running on NPU\n", avg);
    else
        printf("WARN: avg %ldus ≥ 50ms — check for CPU fallback\n", avg);

    face_detector_destroy(fd);
    printf("\nDone.\n");
    return 0;
}
