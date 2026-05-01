#pragma once
#include <stdint.h>

/* ── Parallel 8080 pin mapping ───────────────────────────────────────────────
 *
 *  Data bus  D0-D7  → GPIO 0-7  (must be consecutive starting at 0)
 *  WR        ─────  → GPIO 8    (write strobe, active low)
 *  CS        ─────  → GPIO 9    (chip select, active low)
 *  DC        ─────  → GPIO 10   (0=command, 1=data)
 *  RST       ─────  → 3.3V      (tie high — software reset via cmd 0x01)
 *  RD        ─────  → 3.3V      (tie high — we never read)
 *  BL        ─────  → 3.3V      (backlight hardwired on shield)
 *
 *  Inter-C3 UART (commands from Main C3):
 *  UART1 RX  ─────  → GPIO 20
 *  UART1 TX  ─────  → GPIO 21  (optional, for debug)
 *
 *  NOTE: GPIO 18/19 are USB D-/D+ on most ESP32-C3 boards — do not use.
 */
#define PIN_D0    0
#define PIN_WR    8
#define PIN_CS    9
#define PIN_DC    10

/* ── Display dimensions (landscape) ─────────────────────────────────────── */
#define LCD_W  320
#define LCD_H  240

/* ── Common RGB565 colors ────────────────────────────────────────────────── */
#define COLOR_BLACK    0x0000
#define COLOR_WHITE    0xFFFF
#define COLOR_NAVY     0x000F
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

void ili9341_init(void);
void ili9341_set_window(uint16_t x0, uint16_t y0, uint16_t x1, uint16_t y1);
void ili9341_write_color(uint16_t color, uint32_t count);
void ili9341_write_pixels(const uint16_t *buf, uint32_t count);

/* Streaming API: call set_window once, then stream_pixels for each row,
 * then stream_end().  CS stays LOW between calls — no per-row address overhead. */
void ili9341_stream_pixels(const uint16_t *buf, uint32_t count);
void ili9341_stream_end(void);
