/**
 * gru_policy_runner.h
 *
 * Thin wrapper around cviruntime for running the GRU policy cvimodel
 * on the SG2002 NPU.
 *
 * The GRU hidden state is managed here in CPU memory so the NPU can
 * execute single-step inference while the caller drives the control loop.
 *
 * Usage:
 *   GRURunner *runner = gru_runner_create("gru_policy.cvimodel");
 *   float obs[OBS_SIZE];
 *   GRUOutput out;
 *   while (running) {
 *       read_sensors(obs);
 *       gru_runner_step(runner, obs, &out);
 *       apply_action(out.action);
 *   }
 *   gru_runner_destroy(runner);
 */

#ifndef GRU_POLICY_RUNNER_H
#define GRU_POLICY_RUNNER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Model dimensions (must match the exported ONNX / cvimodel) ─────────── */
#define GRU_INPUT_SIZE   8
#define GRU_HIDDEN_SIZE  64
#define GRU_NUM_LAYERS   1
#define GRU_SEQ_LEN      10
#define GRU_ACTION_SIZE  4

/* ── Output struct ──────────────────────────────────────────────────────── */
typedef struct {
    float  action_logits[GRU_ACTION_SIZE];  /* raw logits */
    int    action;                          /* argmax(logits) */
    float  value;                           /* critic value estimate */
} GRUOutput;

/* ── Opaque runner handle ───────────────────────────────────────────────── */
typedef struct GRURunner GRURunner;

/**
 * Create a runner and load the cvimodel onto the NPU.
 * @param model_path  Path to .cvimodel file
 * @return            Pointer to runner, or NULL on failure
 */
GRURunner *gru_runner_create(const char *model_path);

/**
 * Run one inference step.
 *
 * Internally maintains the GRU hidden state between calls.
 * The observation buffer `obs` must have GRU_INPUT_SIZE floats.
 *
 * @param runner  Runner handle
 * @param obs     Input observation vector [GRU_INPUT_SIZE]
 * @param out     Output struct to fill
 * @return        0 on success, non-zero on error
 */
int gru_runner_step(GRURunner *runner, const float *obs, GRUOutput *out);

/**
 * Reset the hidden state to zeros (call at episode start).
 */
void gru_runner_reset(GRURunner *runner);

/**
 * Destroy the runner and free all resources.
 */
void gru_runner_destroy(GRURunner *runner);

#ifdef __cplusplus
}
#endif

#endif /* GRU_POLICY_RUNNER_H */
