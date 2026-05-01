/*
 * face_detector.c — SCRFD face detection on SG2002 NPU
 *
 * Model output tensor layout (all FP32 after dequant):
 *   score_8_Sigmoid_dequant   [H/8 * W/8 * 2]          confidence per anchor
 *   score_16_Sigmoid_dequant  [H/16 * W/16 * 2]
 *   score_32_Sigmoid_dequant  [H/32 * W/32 * 2]
 *   bbox_8_Conv_dequant       [H/8 * W/8 * 2 * 4]      (dl,dt,dr,db) offsets
 *   bbox_16_Conv_dequant      [H/16 * W/16 * 2 * 4]
 *   bbox_32_Conv_dequant      [H/32 * W/32 * 2 * 4]
 *   kps_8_Conv_dequant        (keypoints — unused by Nexu)
 *   kps_16_Conv_dequant
 *   kps_32_Conv_dequant
 *
 * Anchor generation: 2 anchors per location, generated as square anchors
 *   stride 8:  sizes [16, 32]
 *   stride 16: sizes [64, 128]
 *   stride 32: sizes [256, 512]
 * (standard SCRFD-500M anchor config)
 *
 * BBox decode:
 *   anchor_cx = col * stride + stride/2
 *   anchor_cy = row * stride + stride/2
 *   x1 = anchor_cx - bbox[0] * stride
 *   y1 = anchor_cy - bbox[1] * stride
 *   x2 = anchor_cx + bbox[2] * stride
 *   y2 = anchor_cy + bbox[3] * stride
 */

#include "face_detector.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <cviruntime.h>
#include <cvi_sys.h>

#define TENSOR_STRIDE_BYTES  168   /* CVI_TENSOR struct size */
#define NUM_ANCHORS_PER_LOC  2
#define IMAGE_AREA           ((float)(FACE_MODEL_W * FACE_MODEL_H))

struct FaceDetector {
    CVI_MODEL_HANDLE  model;
    CVI_TENSOR       *inputs;
    CVI_TENSOR       *outputs;
    int32_t           n_in, n_out;

    /* Resolved output tensor pointers */
    float *score[3];   /* stride 8, 16, 32 */
    float *bbox[3];
};

/* ── helpers ──────────────────────────────────────────────────────────────── */

static float clampf(float v, float lo, float hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

static float iou(const FaceBox *a, const FaceBox *b)
{
    float ix1 = a->x1 > b->x1 ? a->x1 : b->x1;
    float iy1 = a->y1 > b->y1 ? a->y1 : b->y1;
    float ix2 = a->x2 < b->x2 ? a->x2 : b->x2;
    float iy2 = a->y2 < b->y2 ? a->y2 : b->y2;
    if (ix2 <= ix1 || iy2 <= iy1) return 0.0f;
    float inter = (ix2 - ix1) * (iy2 - iy1);
    float aa = (a->x2-a->x1)*(a->y2-a->y1);
    float ab = (b->x2-b->x1)*(b->y2-b->y1);
    return inter / (aa + ab - inter + 1e-6f);
}

/* Simple greedy NMS — O(n^2) fine for FACE_MAX_DETECTIONS small */
static int nms(FaceBox *boxes, int n, float thresh)
{
    /* Sort descending by score (bubble sort — n is small) */
    for (int i = 0; i < n-1; i++)
        for (int j = i+1; j < n; j++)
            if (boxes[j].score > boxes[i].score) {
                FaceBox tmp = boxes[i]; boxes[i] = boxes[j]; boxes[j] = tmp;
            }

    int keep = 0;
    int suppressed[FACE_MAX_DETECTIONS] = {0};
    for (int i = 0; i < n; i++) {
        if (suppressed[i]) continue;
        boxes[keep++] = boxes[i];
        for (int j = i+1; j < n; j++)
            if (!suppressed[j] && iou(&boxes[i], &boxes[j]) > thresh)
                suppressed[j] = 1;
    }
    return keep;
}

/* Decode one stride level → append to raw_boxes, return count added */
static int decode_stride(
    const float *scores, const float *bboxes,
    int stride, int rows, int cols,
    FaceBox *raw_boxes, int raw_cap, int raw_n)
{
    for (int row = 0; row < rows && raw_n < raw_cap; row++) {
        for (int col = 0; col < cols && raw_n < raw_cap; col++) {
            for (int a = 0; a < NUM_ANCHORS_PER_LOC && raw_n < raw_cap; a++) {
                int idx = (row * cols + col) * NUM_ANCHORS_PER_LOC + a;
                float score = scores[idx];
                if (score < FACE_SCORE_THRESH) continue;

                float cx = col * stride + stride * 0.5f;
                float cy = row * stride + stride * 0.5f;

                const float *b = &bboxes[idx * 4];
                float x1 = cx - b[0] * stride;
                float y1 = cy - b[1] * stride;
                float x2 = cx + b[2] * stride;
                float y2 = cy + b[3] * stride;

                raw_boxes[raw_n].x1    = clampf(x1, 0, FACE_MODEL_W);
                raw_boxes[raw_n].y1    = clampf(y1, 0, FACE_MODEL_H);
                raw_boxes[raw_n].x2    = clampf(x2, 0, FACE_MODEL_W);
                raw_boxes[raw_n].y2    = clampf(y2, 0, FACE_MODEL_H);
                raw_boxes[raw_n].score = score;
                raw_n++;
            }
        }
    }
    return raw_n;
}

/* ── Public API ───────────────────────────────────────────────────────────── */

FaceDetector *face_detector_create(const char *model_path)
{
    FaceDetector *fd = calloc(1, sizeof(FaceDetector));
    if (!fd) return NULL;

    CVI_RC rc = CVI_NN_RegisterModel(model_path, &fd->model);
    if (rc != CVI_RC_SUCCESS) {
        fprintf(stderr, "[FaceDetector] RegisterModel failed: %d\n", rc);
        free(fd);
        return NULL;
    }

    rc = CVI_NN_GetInputOutputTensors(fd->model,
                                      &fd->inputs,  &fd->n_in,
                                      &fd->outputs, &fd->n_out);
    if (rc != CVI_RC_SUCCESS || fd->n_out < 6) {
        fprintf(stderr, "[FaceDetector] GetTensors failed: rc=%d n_out=%d\n",
                rc, fd->n_out);
        CVI_NN_CleanupModel(fd->model);
        free(fd);
        return NULL;
    }

    /* Map tensors by name (order from model inspector):
     * out[0]=score_8  out[1]=score_16  out[2]=score_32
     * out[3]=bbox_8   out[4]=bbox_16   out[5]=bbox_32   */
    for (int i = 0; i < fd->n_out; i++) {
        const char *name = CVI_NN_TensorName(&fd->outputs[i]);
        float *ptr = (float *)CVI_NN_TensorPtr(&fd->outputs[i]);
        if      (strstr(name, "score_8"))  fd->score[0] = ptr;
        else if (strstr(name, "score_16")) fd->score[1] = ptr;
        else if (strstr(name, "score_32")) fd->score[2] = ptr;
        else if (strstr(name, "bbox_8"))   fd->bbox[0]  = ptr;
        else if (strstr(name, "bbox_16"))  fd->bbox[1]  = ptr;
        else if (strstr(name, "bbox_32"))  fd->bbox[2]  = ptr;
    }

    if (!fd->score[0] || !fd->bbox[0]) {
        fprintf(stderr, "[FaceDetector] Tensor name lookup failed\n");
        CVI_NN_CleanupModel(fd->model);
        free(fd);
        return NULL;
    }

    /* Dump input tensor info to diagnose format/qscale */
    {
        CVI_TENSOR *in = &fd->inputs[0];
        CVI_SHAPE sh = CVI_NN_TensorShape(in);
        printf("[FaceDetector] input[0]: name=%s fmt=%d pixel_fmt=%d qscale=%.6f zp=%d\n"
               "  shape=[%d,%d,%d,%d] dim_size=%zu mem_size=%zu\n"
               "  mean=[%.3f,%.3f,%.3f] scale=[%.6f,%.6f,%.6f]\n",
               CVI_NN_TensorName(in),
               (int)in->fmt, (int)in->pixel_format,
               CVI_NN_TensorQuantScale(in),
               CVI_NN_TensorQuantZeroPoint(in),
               sh.dim[0], sh.dim[1], sh.dim[2], sh.dim[3], sh.dim_size,
               in->mem_size,
               in->mean[0], in->mean[1], in->mean[2],
               in->scale[0], in->scale[1], in->scale[2]);
        printf("[FaceDetector] paddr=0x%lx sys_mem=%p\n",
               (unsigned long)in->paddr, (void*)in->sys_mem);
    }

    printf("[FaceDetector] Ready: %s  (n_in=%d n_out=%d)\n",
           model_path, fd->n_in, fd->n_out);
    return fd;
}

int face_detector_run(FaceDetector *fd,
                      const int8_t *img_bgr_int8,
                      FaceResult   *result)
{
    if (!fd || !result) return -1;

    /* Write image into NPU input tensor */
    int8_t *in_ptr = (int8_t *)CVI_NN_TensorPtr(&fd->inputs[0]);
    size_t in_size = 3 * FACE_MODEL_H * FACE_MODEL_W * sizeof(int8_t);
    if (img_bgr_int8) {
        memcpy(in_ptr, img_bgr_int8, in_size);
    } else {
        memset(in_ptr, 0, in_size);
    }
    /* Flush CPU cache so NPU DMA reads our written data, not stale cache */
    CVI_SYS_IonFlushCache(fd->inputs[0].paddr,
                          (void *)fd->inputs[0].sys_mem,
                          (CVI_U32)fd->inputs[0].mem_size);

    /* Run NPU inference */
    CVI_RC rc = CVI_NN_Forward(fd->model, fd->inputs, fd->n_in,
                               fd->outputs, fd->n_out);
    if (rc != CVI_RC_SUCCESS) {
        fprintf(stderr, "[FaceDetector] Forward failed: %d\n", rc);
        return (int)rc;
    }

    /* Print max scores for diagnostics (remove once working) */
    {
        static int dbg = 0;
        if (dbg < 5) {
            for (int s = 0; s < 3; s++) {
                if (!fd->score[s]) continue;
                int strd = (s==0?8:s==1?16:32);
                int r = FACE_MODEL_H/strd, c = FACE_MODEL_W/strd;
                float mx = -1e9f, mn = 1e9f;
                for (int i = 0; i < r*c*2; i++) {
                    if (fd->score[s][i] > mx) mx = fd->score[s][i];
                    if (fd->score[s][i] < mn) mn = fd->score[s][i];
                }
                printf("[FaceDetector] stride%d score: min=%.4f max=%.4f\n", strd, mn, mx);
            }
            dbg++;
        }
    }

    /* Decode detections from 3 stride levels */
    FaceBox raw[FACE_MAX_DETECTIONS * 4];  /* generously sized before NMS */
    int raw_n = 0;
    int strides[] = {8, 16, 32};
    int rows[]    = {FACE_MODEL_H/8, FACE_MODEL_H/16, FACE_MODEL_H/32};
    int cols[]    = {FACE_MODEL_W/8, FACE_MODEL_W/16, FACE_MODEL_W/32};

    for (int s = 0; s < 3; s++) {
        raw_n = decode_stride(fd->score[s], fd->bbox[s],
                              strides[s], rows[s], cols[s],
                              raw, FACE_MAX_DETECTIONS * 4, raw_n);
    }

    /* NMS */
    int n_kept = nms(raw, raw_n, FACE_NMS_THRESH);
    if (n_kept > FACE_MAX_DETECTIONS) n_kept = FACE_MAX_DETECTIONS;

    memset(result, 0, sizeof(*result));
    result->num_faces = n_kept;

    float max_area = 0.0f;
    for (int i = 0; i < n_kept; i++) {
        result->faces[i] = raw[i];
        float area = (raw[i].x2 - raw[i].x1) * (raw[i].y2 - raw[i].y1);
        if (area > max_area) max_area = area;
    }

    result->person_visible      = (n_kept > 0);
    result->largest_area_norm   = max_area / IMAGE_AREA;
    /* Distance: sqrt of normalized area gives a more linear feel */
    result->person_distance     = sqrtf(result->largest_area_norm);

    return 0;
}

void face_detector_destroy(FaceDetector *fd)
{
    if (!fd) return;
    CVI_NN_CleanupModel(fd->model);
    free(fd);
}
