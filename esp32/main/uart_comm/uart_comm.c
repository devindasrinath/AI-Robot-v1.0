#include "uart_comm.h"

#include <string.h>
#include <stdio.h>
#include "driver/uart.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

static const char *TAG = "uart_comm";

#define RX_BUF_SIZE  512
#define TX_BUF_SIZE  256

static QueueHandle_t s_cmd_queue;   /* → display_task */
static QueueHandle_t s_tx_queue;    /* internal TX queue */

/* ── Parse incoming line ─────────────────────────────────────────────────── */

static void parse_line(const char *line)
{
    UartCmd cmd = { .type = CMD_UNKNOWN, .arg = {0} };

    if (strncmp(line, "DISPLAY:", 8) == 0) {
        cmd.type = CMD_DISPLAY;
        snprintf(cmd.arg, sizeof(cmd.arg), "%.*s", (int)(sizeof(cmd.arg) - 1), line + 8);
    } else if (strncmp(line, "PLAY:", 5) == 0) {
        cmd.type = CMD_PLAY;
        snprintf(cmd.arg, sizeof(cmd.arg), "%.*s", (int)(sizeof(cmd.arg) - 1), line + 5);
    } else {
        ESP_LOGW(TAG, "Unknown command: %s", line);
        return;
    }

    /* Strip trailing whitespace from arg */
    int len = strlen(cmd.arg);
    while (len > 0 && (cmd.arg[len-1] == '\r' || cmd.arg[len-1] == '\n' ||
                       cmd.arg[len-1] == ' '))
        cmd.arg[--len] = '\0';

    xQueueSend(s_cmd_queue, &cmd, 0);
}

/* ── RX task — reads lines from SG2002 ──────────────────────────────────── */

static void uart_rx_task(void *arg)
{
    static char line[128];
    static int  pos = 0;
    uint8_t     ch;

    while (1) {
        int n = uart_read_bytes(UART_SG2002_PORT, &ch, 1, pdMS_TO_TICKS(10));
        if (n <= 0) continue;

        if (ch == '\n') {
            line[pos] = '\0';
            if (pos > 0) parse_line(line);
            pos = 0;
        } else if (ch != '\r') {
            if (pos < (int)sizeof(line) - 1)
                line[pos++] = ch;
        }
    }
}

/* ── TX task — sends events to SG2002 ───────────────────────────────────── */

static void uart_tx_task(void *arg)
{
    char msg[64];
    while (1) {
        if (xQueueReceive(s_tx_queue, msg, portMAX_DELAY) == pdTRUE)
            uart_write_bytes(UART_SG2002_PORT, msg, strlen(msg));
    }
}

/* ── Public API ──────────────────────────────────────────────────────────── */

void uart_send_event(const char *event)
{
    char msg[64];
    snprintf(msg, sizeof(msg), "%s", event);
    /* Non-blocking: drop if queue full */
    xQueueSend(s_tx_queue, msg, 0);
}

void uart_comm_init(QueueHandle_t cmd_queue)
{
    s_cmd_queue = cmd_queue;
    s_tx_queue  = xQueueCreate(8, 64);

    uart_config_t cfg = {
        .baud_rate  = UART_SG2002_BAUD,
        .data_bits  = UART_DATA_8_BITS,
        .parity     = UART_PARITY_DISABLE,
        .stop_bits  = UART_STOP_BITS_1,
        .flow_ctrl  = UART_HW_FLOWCTRL_DISABLE,
    };
    ESP_ERROR_CHECK(uart_param_config(UART_SG2002_PORT, &cfg));
    ESP_ERROR_CHECK(uart_set_pin(UART_SG2002_PORT,
                                 UART_SG2002_TX, UART_SG2002_RX,
                                 UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
    ESP_ERROR_CHECK(uart_driver_install(UART_SG2002_PORT,
                                        RX_BUF_SIZE, TX_BUF_SIZE, 0, NULL, 0));

    xTaskCreate(uart_rx_task, "uart_rx", 2048, NULL, 5, NULL);
    xTaskCreate(uart_tx_task, "uart_tx", 2048, NULL, 5, NULL);

    ESP_LOGI(TAG, "UART1 ready @ %d baud  TX=%d RX=%d",
             UART_SG2002_BAUD, UART_SG2002_TX, UART_SG2002_RX);
}
