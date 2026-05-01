#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_log.h"

#include "display/st7789.h"
#include "display/gfx.h"
#include "display/face.h"
#include "uart_comm/uart_comm.h"
#include "sensors/touch.h"
#include "audio/i2s_audio.h"

static const char *TAG = "nexu";

/* Queues */
static QueueHandle_t s_cmd_queue;    /* uart_rx_task → display_task */
static QueueHandle_t s_audio_queue;  /* display_task → audio_task   */

/* ── Display task ────────────────────────────────────────────────────────── */

static void display_task(void *arg)
{
    /* Display test — red → green → blue, 600ms each. Remove once verified. */
    st7789_set_window(0, 0, LCD_W - 1, LCD_H - 1);
    st7789_write_color(RGB(255, 0, 0), LCD_W * LCD_H);
    vTaskDelay(pdMS_TO_TICKS(600));
    st7789_set_window(0, 0, LCD_W - 1, LCD_H - 1);
    st7789_write_color(RGB(0, 255, 0), LCD_W * LCD_H);
    vTaskDelay(pdMS_TO_TICKS(600));
    st7789_set_window(0, 0, LCD_W - 1, LCD_H - 1);
    st7789_write_color(RGB(0, 0, 255), LCD_W * LCD_H);
    vTaskDelay(pdMS_TO_TICKS(600));

    /* Boot: init animator, start with idle */
    face_init();
    ESP_LOGI(TAG, "Display ready — face animator running");

    while (1) {
        /* Process any pending display command (non-blocking) */
        UartCmd cmd;
        if (xQueueReceive(s_cmd_queue, &cmd, 0) == pdTRUE) {
            if (cmd.type == CMD_DISPLAY) {
                FaceExpression expr = face_from_string(cmd.arg);
                ESP_LOGI(TAG, "Display: %s (%d)", cmd.arg, expr);
                face_set(expr);   /* smooth transition, not instant jump */
            } else if (cmd.type == CMD_PLAY) {
                xQueueSend(s_audio_queue, cmd.arg, 0);
            }
        }

        /* Advance face animation — always yield after draw so IDLE can run */
        face_tick();
        vTaskDelay(pdMS_TO_TICKS(33));  /* ~30fps — SPI DMA needs the headroom */
    }
}

/* ── app_main ────────────────────────────────────────────────────────────── */

void app_main(void)
{
    ESP_LOGI(TAG, "Nexu ESP32-C3 booting...");

    /* 1. Display hardware init */
    st7789_init();

    /* 2. Command queue (uart → display) */
    s_cmd_queue = xQueueCreate(8, sizeof(UartCmd));

    /* 3. Audio queue (display_task → audio_task), items are char[32] names */
    s_audio_queue = xQueueCreate(4, 32);

    /* 4. UART comm with SG2002 */
    uart_comm_init(s_cmd_queue);

    /* 5. Touch sensor */
    touch_init();

    /* 6. Audio — I2S + PCM5102 + tone synthesizer */
    audio_init(s_audio_queue);

    /* 7. Display task — highest priority for smooth animation */
    xTaskCreate(display_task, "display", 4096, NULL, 4, NULL);

    ESP_LOGI(TAG, "All systems go. Nexu is alive.");
}
