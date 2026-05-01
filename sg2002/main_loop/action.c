#include "action.h"
#include "uart_sg.h"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

const char *ACTION_NAMES[NUM_ACTIONS] = {
    "idle_sit", "look_around", "approach",  "follow",
    "play_gesture", "vocal_happy", "vocal_curious", "vocal_alert",
    "sleep", "avoid_obstacle", "self_groom", "wag_tail"
};

/* ── Temperature sampling ────────────────────────────────────────────────── */

int action_sample(const float *logits, float temperature)
{
    float probs[NUM_ACTIONS];
    float sum = 0.0f;

    for (int i = 0; i < NUM_ACTIONS; i++) {
        probs[i] = expf(logits[i] / temperature);
        sum += probs[i];
    }

    float r = ((float)rand() / RAND_MAX) * sum;
    float cumsum = 0.0f;
    for (int i = 0; i < NUM_ACTIONS; i++) {
        cumsum += probs[i];
        if (r <= cumsum) return i;
    }
    return NUM_ACTIONS - 1;
}

/* ── Action → UART command mapping ──────────────────────────────────────── */

void action_dispatch(int action)
{
    switch (action) {
    case 0:  /* idle_sit */
        uart_sg_send("DISPLAY:idle\n");
        break;
    case 1:  /* look_around */
        uart_sg_send("DISPLAY:curious\n");
        break;
    case 2:  /* approach — face curious, motors: future */
        uart_sg_send("DISPLAY:curious\n");
        break;
    case 3:  /* follow */
        uart_sg_send("DISPLAY:happy\n");
        break;
    case 4:  /* play_gesture */
        uart_sg_send("DISPLAY:happy\n");
        uart_sg_send("PLAY:vocal_play.wav\n");
        break;
    case 5:  /* vocal_happy */
        uart_sg_send("DISPLAY:happy\n");
        uart_sg_send("PLAY:vocal_happy.wav\n");
        break;
    case 6:  /* vocal_curious */
        uart_sg_send("DISPLAY:curious\n");
        uart_sg_send("PLAY:vocal_curious.wav\n");
        break;
    case 7:  /* vocal_alert */
        uart_sg_send("DISPLAY:alert\n");
        uart_sg_send("PLAY:vocal_alert.wav\n");
        break;
    case 8:  /* sleep */
        uart_sg_send("DISPLAY:sleep\n");
        break;
    case 9:  /* avoid_obstacle — motors: future */
        uart_sg_send("DISPLAY:alert\n");
        break;
    case 10: /* self_groom */
        uart_sg_send("DISPLAY:idle\n");
        break;
    case 11: /* wag_tail */
        uart_sg_send("DISPLAY:happy\n");
        break;
    default:
        uart_sg_send("DISPLAY:idle\n");
        break;
    }

    printf("[action] %s\n", ACTION_NAMES[action]);
}
