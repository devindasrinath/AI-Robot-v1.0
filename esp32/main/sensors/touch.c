#include "touch.h"
#include "uart_comm.h"

#include "driver/gpio.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "touch";

#define DEBOUNCE_MS   80    /* ignore re-triggers within 80ms */
#define COOLDOWN_MS   500   /* minimum gap between TOUCH events to SG2002 */

static void touch_task(void *arg)
{
    gpio_config_t io = {
        .pin_bit_mask = 1ULL << PIN_TOUCH_HEAD,
        .mode         = GPIO_MODE_INPUT,
        .pull_up_en   = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_ENABLE,
    };
    gpio_config(&io);

    bool     last_state  = false;
    uint32_t last_event  = 0;
    uint32_t last_debounce = 0;

    ESP_LOGI(TAG, "Touch sensor ready on GPIO%d", PIN_TOUCH_HEAD);

    while (1) {
        vTaskDelay(pdMS_TO_TICKS(10));

        bool state = gpio_get_level(PIN_TOUCH_HEAD);
        uint32_t now = xTaskGetTickCount() * portTICK_PERIOD_MS;

        /* Debounce: ignore transitions faster than DEBOUNCE_MS */
        if (state != last_state) {
            last_debounce = now;
            last_state = state;
        }

        if (state && (now - last_debounce > DEBOUNCE_MS)) {
            /* Confirmed touch — send event with cooldown */
            if (now - last_event > COOLDOWN_MS) {
                ESP_LOGI(TAG, "Head touch!");
                uart_send_event("TOUCH:head\n");
                last_event = now;
            }
        }
    }
}

void touch_init(void)
{
    xTaskCreate(touch_task, "touch", 2048, NULL, 4, NULL);
}
