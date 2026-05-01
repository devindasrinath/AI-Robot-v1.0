#pragma once
#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"

/* ── Pin config ──────────────────────────────────────────────────────────── */
#define UART_SG2002_PORT   UART_NUM_1
#define UART_SG2002_TX     21
#define UART_SG2002_RX     20
#define UART_SG2002_BAUD   921600

/* ── Command types received from SG2002 ─────────────────────────────────── */
typedef enum {
    CMD_DISPLAY,   /* DISPLAY:<expr>  */
    CMD_PLAY,      /* PLAY:<filename> */
    CMD_UNKNOWN,
} CmdType;

typedef struct {
    CmdType type;
    char    arg[32];   /* expression name or filename */
} UartCmd;

/* ── Events sent to SG2002 ───────────────────────────────────────────────── */
void uart_send_event(const char *event);  /* e.g. "TOUCH:head\n" */

/* ── Init: creates uart_rx_task and uart_tx_task ─────────────────────────── */
void uart_comm_init(QueueHandle_t cmd_queue);
