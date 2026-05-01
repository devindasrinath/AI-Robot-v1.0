#pragma once
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"

#define PIN_TOUCH_HEAD   2   /* TTP223 output → GPIO2 */

/* Start touch sensor task. Events sent as "TOUCH:head\n" to uart_send_event. */
void touch_init(void);
