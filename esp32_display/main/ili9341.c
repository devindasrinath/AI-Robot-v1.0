#include "ili9341.h"

#include <string.h>
#include "driver/gpio.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_attr.h"
#include "soc/gpio_reg.h"  /* GPIO_OUT_W1TS_REG, GPIO_OUT_W1TC_REG */

/* ── Bit masks (all GPIOs < 32, single OUT register) ─────────────────────── */
#define DATA_MASK  0x000000FFu          /* GPIO 0-7 */
#define WR_MASK    (1u << PIN_WR)       /* GPIO 8  */
#define CS_MASK    (1u << PIN_CS)       /* GPIO 9  */
#define DC_MASK    (1u << PIN_DC)       /* GPIO 10 */

/* ── Fast inline GPIO macros ─────────────────────────────────────────────── */
#define CS_LOW()   REG_WRITE(GPIO_OUT_W1TC_REG, CS_MASK)
#define CS_HIGH()  REG_WRITE(GPIO_OUT_W1TS_REG, CS_MASK)
#define DC_CMD()   REG_WRITE(GPIO_OUT_W1TC_REG, DC_MASK)
#define DC_DATA()  REG_WRITE(GPIO_OUT_W1TS_REG, DC_MASK)

/* Write one byte to the parallel bus (CS/DC must be set by caller).
 * At 160MHz each REG_WRITE ≈ 12.5ns (APB). Three NOPs ensure WR pulse
 * width ≥ 18ns, safely above ILI9341's 15ns minimum. */
static inline void IRAM_ATTR write_byte(uint8_t b)
{
    REG_WRITE(GPIO_OUT_W1TC_REG, DATA_MASK);        /* clear D0-D7  */
    REG_WRITE(GPIO_OUT_W1TS_REG, (uint32_t)b);      /* set new data */
    REG_WRITE(GPIO_OUT_W1TC_REG, WR_MASK);          /* WR low       */
    __asm__ volatile("nop\nnop\nnop");
    REG_WRITE(GPIO_OUT_W1TS_REG, WR_MASK);          /* WR high      */
    __asm__ volatile("nop\nnop\nnop");
}

/* ── Init helpers ────────────────────────────────────────────────────────── */

static void send_cmd(uint8_t cmd)
{
    CS_LOW(); DC_CMD();
    write_byte(cmd);
    CS_HIGH();
}

static void send_data(const uint8_t *data, size_t len)
{
    CS_LOW(); DC_DATA();
    for (size_t i = 0; i < len; i++) write_byte(data[i]);
    CS_HIGH();
}

static void send_cmd_data(uint8_t cmd, const uint8_t *data, size_t len)
{
    CS_LOW();
    DC_CMD();  write_byte(cmd);
    DC_DATA();
    for (size_t i = 0; i < len; i++) write_byte(data[i]);
    CS_HIGH();
}

/* ── ILI9341 init sequence (Adafruit-derived, well-tested) ──────────────── */

static void ili9341_send_init(void)
{
    /* Software reset — RST is tied to 3.3V so we reset via command */
    send_cmd(0x01);
    vTaskDelay(pdMS_TO_TICKS(150));
    send_cmd_data(0xEF, (uint8_t[]){0x03,0x80,0x02}, 3);
    send_cmd_data(0xCF, (uint8_t[]){0x00,0xC1,0x30}, 3);
    send_cmd_data(0xED, (uint8_t[]){0x64,0x03,0x12,0x81}, 4);
    send_cmd_data(0xE8, (uint8_t[]){0x85,0x00,0x78}, 3);
    send_cmd_data(0xCB, (uint8_t[]){0x39,0x2C,0x00,0x34,0x02}, 5);
    send_cmd_data(0xF7, (uint8_t[]){0x20}, 1);
    send_cmd_data(0xEA, (uint8_t[]){0x00,0x00}, 2);

    send_cmd_data(0xC0, (uint8_t[]){0x23}, 1);           /* PWCTR1  */
    send_cmd_data(0xC1, (uint8_t[]){0x10}, 1);           /* PWCTR2  */
    send_cmd_data(0xC5, (uint8_t[]){0x3e,0x28}, 2);      /* VMCTR1  */
    send_cmd_data(0xC7, (uint8_t[]){0x86}, 1);           /* VMCTR2  */

    /* MADCTL: MV=1 (transpose) + BGR=1 — landscape 320×240.
     * If image appears upside-down, change 0x28 → 0xE8. */
    send_cmd_data(0x36, (uint8_t[]){0x28}, 1);

    send_cmd_data(0x3A, (uint8_t[]){0x55}, 1);           /* COLMOD: 16-bit RGB565 */
    send_cmd_data(0xB1, (uint8_t[]){0x00,0x18}, 2);      /* FRMCTR1: 60Hz         */
    send_cmd_data(0xB6, (uint8_t[]){0x08,0x82,0x27}, 3); /* DFUNCTR               */
    send_cmd_data(0xF2, (uint8_t[]){0x00}, 1);           /* 3G off                */
    send_cmd_data(0x26, (uint8_t[]){0x01}, 1);           /* Gamma set 1           */

    send_cmd_data(0xE0,
        (uint8_t[]){0x0F,0x31,0x2B,0x0C,0x0E,0x08,
                    0x4E,0xF1,0x37,0x07,0x10,0x03,
                    0x0E,0x09,0x00}, 15);                 /* Positive gamma        */

    send_cmd_data(0xE1,
        (uint8_t[]){0x00,0x0E,0x14,0x03,0x11,0x07,
                    0x31,0xC1,0x48,0x08,0x0F,0x0C,
                    0x31,0x36,0x0F}, 15);                 /* Negative gamma        */

    send_cmd(0x11);                                       /* Sleep out             */
    vTaskDelay(pdMS_TO_TICKS(120));
    send_cmd(0x29);                                       /* Display on            */
    vTaskDelay(pdMS_TO_TICKS(20));
}

/* ── Public API ──────────────────────────────────────────────────────────── */

void ili9341_init(void)
{
    /* Configure all output GPIOs (RST tied to 3.3V — no GPIO needed) */
    uint64_t pin_mask = DATA_MASK | WR_MASK | CS_MASK | DC_MASK;
    gpio_config_t io = {
        .pin_bit_mask = pin_mask,
        .mode         = GPIO_MODE_OUTPUT,
        .pull_up_en   = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type    = GPIO_INTR_DISABLE,
    };
    gpio_config(&io);

    /* Safe idle state: CS high, WR high */
    REG_WRITE(GPIO_OUT_W1TS_REG, CS_MASK | WR_MASK);

    ili9341_send_init();
}

void IRAM_ATTR ili9341_set_window(uint16_t x0, uint16_t y0,
                                   uint16_t x1, uint16_t y1)
{
    uint8_t ca[4] = { x0>>8, x0&0xFF, x1>>8, x1&0xFF };
    uint8_t ra[4] = { y0>>8, y0&0xFF, y1>>8, y1&0xFF };

    CS_LOW();
    DC_CMD(); write_byte(0x2A);   /* Column address set */
    DC_DATA();
    for (int i = 0; i < 4; i++) write_byte(ca[i]);

    DC_CMD(); write_byte(0x2B);   /* Row address set    */
    DC_DATA();
    for (int i = 0; i < 4; i++) write_byte(ra[i]);

    DC_CMD(); write_byte(0x2C);   /* Memory write       */
    DC_DATA();
    /* CS stays LOW — caller writes pixels immediately after */
}

void IRAM_ATTR ili9341_write_color(uint16_t color, uint32_t count)
{
    uint8_t hi = color >> 8;
    uint8_t lo = color & 0xFF;
    while (count--) {
        write_byte(hi);
        write_byte(lo);
    }
    CS_HIGH();
}

void IRAM_ATTR ili9341_write_pixels(const uint16_t *buf, uint32_t count)
{
    while (count--) {
        write_byte((*buf) >> 8);
        write_byte((*buf) & 0xFF);
        buf++;
    }
    CS_HIGH();
}

/* Streaming API: set_window once, then stream_pixels per row, then stream_end.
 * CS stays LOW the entire time — saves per-row address overhead (11 bytes/row). */
void IRAM_ATTR ili9341_stream_pixels(const uint16_t *buf, uint32_t count)
{
    while (count--) {
        write_byte((*buf) >> 8);
        write_byte((*buf) & 0xFF);
        buf++;
    }
    /* CS stays LOW intentionally */
}

void IRAM_ATTR ili9341_stream_end(void)
{
    CS_HIGH();
}
