#pragma once
#include <string.h>

#define OBS_SIZE    16
#define NUM_ACTIONS 12
#define HIDDEN_SIZE 64
#define MEL_ELEMS   (40 * 98)   /* wake word input: 40 mel bins × 98 frames */

/* Observation vector — fed into GRU every 100ms */
typedef struct {
    float touch_head;           /* [0]  1.0 = touched this tick */
    float touch_body;           /* [1] */
    float touch_tail;           /* [2] */
    float collision_front;      /* [3]  1.0 = obstacle detected */
    float collision_left;       /* [4] */
    float collision_right;      /* [5] */
    float person_visible;       /* [6]  1.0 = face detected in frame */
    float person_distance;      /* [7]  1.0 = very close, 0.0 = far/absent */
    float wake_word;            /* [8]  1.0 = "Nexu" detected this tick */
    float battery_level;        /* [9]  1.0 = full (stub: fixed at 0.95) */
    float time_since_touch;     /* [10] 0→1 over ~60s since last touch */
    float idle_ticks_norm;      /* [11] 0→1 over ~300 idle ticks */
    float sound_level;          /* [12] stub: 0 */
    float motion_energy;        /* [13] stub: 0 */
    float personality_curiosity;/* [14] randomised at boot, fixed per session */
    float personality_playful;  /* [15] randomised at boot, fixed per session */
} ObsVector;

static inline void obs_to_array(const ObsVector *obs, float *arr)
{
    arr[0]  = obs->touch_head;
    arr[1]  = obs->touch_body;
    arr[2]  = obs->touch_tail;
    arr[3]  = obs->collision_front;
    arr[4]  = obs->collision_left;
    arr[5]  = obs->collision_right;
    arr[6]  = obs->person_visible;
    arr[7]  = obs->person_distance;
    arr[8]  = obs->wake_word;
    arr[9]  = obs->battery_level;
    arr[10] = obs->time_since_touch;
    arr[11] = obs->idle_ticks_norm;
    arr[12] = obs->sound_level;
    arr[13] = obs->motion_energy;
    arr[14] = obs->personality_curiosity;
    arr[15] = obs->personality_playful;
}
