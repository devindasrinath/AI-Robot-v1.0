#pragma once
/*
 * face_detector.h — SCRFD face detection on SG2002 NPU
 *
 * Model: scrfd_768_432_int8_1x.cvimodel
 * Input:  [1, 3, 432, 768] INT8 (BGR, NCHW)
 * Output: 9 tensors (score+bbox+kps at stride 8/16/32)
 *
 * For Nexu:
 *   person_visible  = any detection above threshold
 *   person_distance = largest_face_area / image_area (0=far, 1=very close)
 */

#include <stdint.h>
#include <stdbool.h>

#define FACE_MAX_DETECTIONS  16
#define FACE_MODEL_W         768
#define FACE_MODEL_H         432
#define FACE_SCORE_THRESH    0.5f
#define FACE_NMS_THRESH      0.4f

typedef struct {
    float x1, y1, x2, y2;   /* bbox in model pixel coords */
    float score;
} FaceBox;

typedef struct {
    int      num_faces;
    FaceBox  faces[FACE_MAX_DETECTIONS];
    float    largest_area_norm;  /* largest face bbox area / (W*H), 0..1 */
    bool     person_visible;
    float    person_distance;    /* 0=far/none, 1=very close */
} FaceResult;

typedef struct FaceDetector FaceDetector;

/* Create detector — loads cvimodel onto NPU. Returns NULL on failure. */
FaceDetector *face_detector_create(const char *model_path);

/*
 * Run inference.
 * img_bgr_int8: [3 * FACE_MODEL_H * FACE_MODEL_W] INT8 planar BGR
 *               Pass NULL to run with zeroed input (latency check only).
 * result: filled with detections.
 * Returns 0 on success.
 */
int face_detector_run(FaceDetector *fd,
                      const int8_t *img_bgr_int8,
                      FaceResult   *result);

void face_detector_destroy(FaceDetector *fd);
