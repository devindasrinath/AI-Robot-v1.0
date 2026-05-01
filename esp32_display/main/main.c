#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"

#include "ili9341.h"
#include "gfx.h"
#include "face.h"
#include "uart_cmd.h"

static const char *TAG = "nexu_display";

/* ── Display task ────────────────────────────────────────────────────────── */

static void display_task(void *arg)
{
    /* Boot color test: red → green → blue (proves display is alive) */
    ili9341_set_window(0, 0, LCD_W - 1, LCD_H - 1);
    ili9341_write_color(RGB(255, 0, 0), LCD_W * LCD_H);
    vTaskDelay(pdMS_TO_TICKS(500));
    ili9341_set_window(0, 0, LCD_W - 1, LCD_H - 1);
    ili9341_write_color(RGB(0, 255, 0), LCD_W * LCD_H);
    vTaskDelay(pdMS_TO_TICKS(500));
    ili9341_set_window(0, 0, LCD_W - 1, LCD_H - 1);
    ili9341_write_color(RGB(0, 0, 255), LCD_W * LCD_H);
    vTaskDelay(pdMS_TO_TICKS(500));

    face_init();
    ESP_LOGI(TAG, "face animator running");

    /* Self-test: cycle expressions every 2s until Main C3 sends commands.
     * This lets you verify display + face rendering without any wiring to
     * the Main C3. Remove or gate behind a flag once integration is done. */
    static const FaceExpression cycle[] = {
        FACE_IDLE, FACE_HAPPY, FACE_CURIOUS, FACE_ALERT, FACE_SLEEP
    };
    int cycle_idx = 0;
    uint32_t last_cycle = 0;

    while (1) {
        uint32_t now = xTaskGetTickCount();
        if (now - last_cycle >= pdMS_TO_TICKS(2000)) {
            face_set(cycle[cycle_idx]);
            cycle_idx = (cycle_idx + 1) % 5;
            last_cycle = now;
        }

        face_tick();
        vTaskDelay(pdMS_TO_TICKS(33));   /* ~30fps — parallel bus is slower than SPI */
    }
}

/* ── app_main ────────────────────────────────────────────────────────────── */

void app_main(void)
{
    ESP_LOGI(TAG, "Nexu Display C3 booting...");

    /* ILI9341 parallel 8080 init */
    ili9341_init();

    /* Inter-C3 UART command receiver */
    uart_cmd_init();

    /* Display task at high priority for smooth animation */
    xTaskCreate(display_task, "display", 4096, NULL, 4, NULL);

    ESP_LOGI(TAG, "ready.");
}
