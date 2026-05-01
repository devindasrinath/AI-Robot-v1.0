#include "st7789.h"

#include <string.h>
#include "driver/spi_master.h"
#include "driver/gpio.h"
#include "driver/ledc.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "st7789";
static spi_device_handle_t s_spi;

/* ── Low-level SPI helpers ───────────────────────────────────────────────── */

static void dc_cmd(void)  { gpio_set_level(PIN_DC, 0); }
static void dc_data(void) { gpio_set_level(PIN_DC, 1); }

static void spi_write_byte(uint8_t b)
{
    spi_transaction_t t = {
        .length    = 8,
        .tx_buffer = &b,
        .flags     = 0,
    };
    spi_device_polling_transmit(s_spi, &t);
}

static void spi_write_buf(const void *buf, size_t len)
{
    if (len == 0) return;
    spi_transaction_t t = {
        .length    = len * 8,
        .tx_buffer = buf,
        .flags     = 0,
    };
    spi_device_polling_transmit(s_spi, &t);
}

static void write_cmd(uint8_t cmd)
{
    dc_cmd();
    spi_write_byte(cmd);
}

static void write_data8(uint8_t d)
{
    dc_data();
    spi_write_byte(d);
}

/* ── ST7789V init sequence ───────────────────────────────────────────────── */

static void st7789_reset(void)
{
    gpio_set_level(PIN_RES, 0);
    vTaskDelay(pdMS_TO_TICKS(10));
    gpio_set_level(PIN_RES, 1);
    vTaskDelay(pdMS_TO_TICKS(120));
}

static void st7789_send_init(void)
{
    write_cmd(0x01);                    /* SW reset */
    vTaskDelay(pdMS_TO_TICKS(150));

    write_cmd(0x11);                    /* Sleep out */
    vTaskDelay(pdMS_TO_TICKS(120));

    write_cmd(0x3A);  write_data8(0x55); /* Color mode: 16-bit RGB565 */
    write_cmd(0x36);  write_data8(0x00); /* MADCTL: normal orientation */

    /* Porch setting */
    write_cmd(0xB2);
    write_data8(0x0C); write_data8(0x0C); write_data8(0x00);
    write_data8(0x33); write_data8(0x33);

    write_cmd(0xB7);  write_data8(0x35); /* Gate control */
    write_cmd(0xBB);  write_data8(0x19); /* VCOMS */
    write_cmd(0xC0);  write_data8(0x2C); /* LCM control */
    write_cmd(0xC2);  write_data8(0x01); /* VDV/VRH enable */
    write_cmd(0xC3);  write_data8(0x12); /* VRH set */
    write_cmd(0xC4);  write_data8(0x20); /* VDV set */
    write_cmd(0xC6);  write_data8(0x0F); /* Frame rate: 60Hz */

    write_cmd(0xD0);  write_data8(0xA4); write_data8(0xA1); /* Power control */

    /* Positive gamma */
    write_cmd(0xE0);
    write_data8(0xD0); write_data8(0x04); write_data8(0x0D); write_data8(0x11);
    write_data8(0x13); write_data8(0x2B); write_data8(0x3F); write_data8(0x54);
    write_data8(0x4C); write_data8(0x18); write_data8(0x0D); write_data8(0x0B);
    write_data8(0x1F); write_data8(0x23);

    /* Negative gamma */
    write_cmd(0xE1);
    write_data8(0xD0); write_data8(0x04); write_data8(0x0C); write_data8(0x11);
    write_data8(0x13); write_data8(0x2C); write_data8(0x3F); write_data8(0x44);
    write_data8(0x51); write_data8(0x2F); write_data8(0x1F); write_data8(0x1F);
    write_data8(0x20); write_data8(0x23);

    write_cmd(0x21);                    /* Display inversion on (needed for ST7789) */
    write_cmd(0x29);                    /* Display on */
    vTaskDelay(pdMS_TO_TICKS(20));
}

/* ── Public API ──────────────────────────────────────────────────────────── */

void st7789_init(void)
{
    /* Configure DC and RES GPIOs */
    gpio_config_t io = {
        .pin_bit_mask = (1ULL << PIN_DC) | (1ULL << PIN_RES),
        .mode         = GPIO_MODE_OUTPUT,
    };
    gpio_config(&io);

    /* SPI bus */
    spi_bus_config_t bus = {
        .mosi_io_num     = PIN_SDA,
        .miso_io_num     = -1,
        .sclk_io_num     = PIN_SCL,
        .quadwp_io_num   = -1,
        .quadhd_io_num   = -1,
        .max_transfer_sz = LCD_W * LCD_H * 2,
    };
    ESP_ERROR_CHECK(spi_bus_initialize(SPI2_HOST, &bus, SPI_DMA_CH_AUTO));

    /* SPI device */
    spi_device_interface_config_t dev = {
        .clock_speed_hz = 40 * 1000 * 1000,   /* 40 MHz */
        .mode           = 0,
        .spics_io_num   = PIN_CS,
        .queue_size     = 7,
    };
    ESP_ERROR_CHECK(spi_bus_add_device(SPI2_HOST, &dev, &s_spi));

    /* Backlight via LEDC PWM */
    ledc_timer_config_t ltimer = {
        .speed_mode      = LEDC_LOW_SPEED_MODE,
        .timer_num       = LEDC_TIMER_0,
        .duty_resolution = LEDC_TIMER_8_BIT,
        .freq_hz         = 5000,
        .clk_cfg         = LEDC_AUTO_CLK,
    };
    ledc_timer_config(&ltimer);

    ledc_channel_config_t lchan = {
        .gpio_num   = PIN_BLK,
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .channel    = LEDC_CHANNEL_0,
        .timer_sel  = LEDC_TIMER_0,
        .duty       = 200,   /* ~78% brightness */
        .hpoint     = 0,
    };
    ledc_channel_config(&lchan);

    st7789_reset();
    st7789_send_init();

    ESP_LOGI(TAG, "ST7789V ready (%dx%d)", LCD_W, LCD_H);
}

void st7789_set_window(uint16_t x0, uint16_t y0, uint16_t x1, uint16_t y1)
{
    write_cmd(0x2A);  /* Column address set */
    dc_data();
    uint8_t ca[4] = { x0>>8, x0&0xFF, x1>>8, x1&0xFF };
    spi_write_buf(ca, 4);

    write_cmd(0x2B);  /* Row address set */
    dc_data();
    uint8_t ra[4] = { y0>>8, y0&0xFF, y1>>8, y1&0xFF };
    spi_write_buf(ra, 4);

    write_cmd(0x2C);  /* Memory write */
    dc_data();
}

void st7789_write_color(uint16_t color, uint32_t count)
{
    /* Swap bytes for SPI (big-endian) */
    uint16_t c = (color >> 8) | (color << 8);

    /* Send in chunks to avoid huge stack allocation */
    static uint16_t chunk[256];
    for (int i = 0; i < 256; i++) chunk[i] = c;

    while (count > 0) {
        uint32_t n = count > 256 ? 256 : count;
        spi_write_buf(chunk, n * 2);
        count -= n;
    }
}

void st7789_write_pixels(const uint16_t *buf, uint32_t count)
{
    /* Caller must byte-swap if needed */
    spi_write_buf(buf, count * 2);
}

void st7789_backlight(uint8_t brightness)
{
    ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0, brightness);
    ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0);
}
