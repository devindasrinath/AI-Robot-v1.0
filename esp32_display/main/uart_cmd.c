#include "uart_cmd.h"
#include "face.h"

#include <string.h>
#include "driver/uart.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"

static const char *TAG = "uart_cmd";

static void uart_rx_task(void *arg)
{
    char buf[64];
    int  pos = 0;

    while (1) {
        uint8_t c;
        int n = uart_read_bytes(UART_CMD_PORT, &c, 1, pdMS_TO_TICKS(100));
        if (n <= 0) continue;

        if (c == '\n' || c == '\r') {
            if (pos == 0) continue;
            buf[pos] = '\0';
            pos = 0;

            /* Parse "DISPLAY:<expr>" */
            if (strncmp(buf, "DISPLAY:", 8) == 0) {
                FaceExpression expr = face_from_string(buf + 8);
                ESP_LOGI(TAG, "cmd: %s → expr %d", buf + 8, (int)expr);
                face_set(expr);
            }
        } else {
            if (pos < (int)sizeof(buf) - 2)
                buf[pos++] = (char)c;
        }
    }
}

void uart_cmd_init(void)
{
    uart_config_t cfg = {
        .baud_rate  = UART_CMD_BAUD,
        .data_bits  = UART_DATA_8_BITS,
        .parity     = UART_PARITY_DISABLE,
        .stop_bits  = UART_STOP_BITS_1,
        .flow_ctrl  = UART_HW_FLOWCTRL_DISABLE,
    };
    ESP_ERROR_CHECK(uart_driver_install(UART_CMD_PORT, 256, 0, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_param_config(UART_CMD_PORT, &cfg));
    ESP_ERROR_CHECK(uart_set_pin(UART_CMD_PORT,
                                 UART_CMD_TX, UART_CMD_RX,
                                 UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));

    xTaskCreate(uart_rx_task, "uart_rx", 2048, NULL, 3, NULL);
    ESP_LOGI(TAG, "inter-C3 UART ready (RX=GPIO%d, %dbaud)",
             UART_CMD_RX, UART_CMD_BAUD);
}
