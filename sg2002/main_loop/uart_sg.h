#pragma once
#include <stdbool.h>

/* UART port connecting SG2002 to ESP32-C3.
   Check /dev/ttyS* on the board — adjust if needed. */
#define UART_ESP_DEV  "/dev/ttyS1"
#define UART_ESP_BAUD 921600

/* Events received from ESP32-C3 (set by RX thread, cleared by main loop) */
typedef struct {
    volatile bool touch_head;
    volatile bool touch_body;
    volatile bool touch_tail;
    volatile bool collision_front;
    volatile bool collision_left;
    volatile bool collision_right;
} UartEvents;

extern UartEvents g_uart_events;

/* Open UART and start RX thread. Returns 0 on success. */
int  uart_sg_init(void);

/* Send command to ESP32-C3, e.g. "DISPLAY:happy\n" */
void uart_sg_send(const char *cmd);
