#pragma once
#include <stdint.h>

/* ── Pin mapping ─────────────────────────────────────────────────────────── */
#define PIN_SCL   4    /* SPI clock  */
#define PIN_SDA   6    /* SPI MOSI   */
#define PIN_RES   8    /* Reset      */
#define PIN_DC    9    /* Data/Cmd   */
#define PIN_CS    5    /* Chip select*/
#define PIN_BLK   7    /* Backlight  */

/* ── Display dimensions ──────────────────────────────────────────────────── */
#define LCD_W     240
#define LCD_H     320

/* ── Common RGB565 colors ────────────────────────────────────────────────── */
#define COLOR_BLACK    0x0000
#define COLOR_WHITE    0xFFFF
#define COLOR_NAVY     0x000F
#define COLOR_DARKBLUE 0x0003
#define COLOR_RED      0xF800
#define COLOR_GREEN    0x07E0
#define COLOR_YELLOW   0xFFE0
#define COLOR_ORANGE   0xFD20
#define COLOR_PINK     0xF81F
#define COLOR_CYAN     0x07FF
#define COLOR_GRAY     0x7BEF
#define COLOR_DARKGRAY 0x39E7

/* Pack RGB888 → RGB565 */
#define RGB(r,g,b) ((uint16_t)( (((r)&0xF8)<<8) | (((g)&0xFC)<<3) | ((b)>>3) ))

void     st7789_init(void);
void     st7789_set_window(uint16_t x0, uint16_t y0, uint16_t x1, uint16_t y1);
void     st7789_write_color(uint16_t color, uint32_t count);
void     st7789_write_pixels(const uint16_t *buf, uint32_t count);
void     st7789_backlight(uint8_t brightness);  /* 0–255 */
