#include "gfx.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

static void draw_hline(int16_t x, int16_t y, int16_t len, uint16_t color)
{
    if (y < 0 || y >= LCD_H || len <= 0) return;
    if (x < 0) { len += x; x = 0; }
    if (x + len > LCD_W) len = LCD_W - x;
    if (len <= 0) return;
    ili9341_set_window(x, y, x + len - 1, y);
    ili9341_write_color(color, len);
}

void gfx_fill_screen(uint16_t color)
{
    ili9341_set_window(0, 0, LCD_W - 1, LCD_H - 1);
    ili9341_write_color(color, (uint32_t)LCD_W * LCD_H);
}

void gfx_fill_rect(int16_t x, int16_t y, int16_t w, int16_t h, uint16_t color)
{
    if (x >= LCD_W || y >= LCD_H || w <= 0 || h <= 0) return;
    if (x < 0) { w += x; x = 0; }
    if (y < 0) { h += y; y = 0; }
    if (x + w > LCD_W) w = LCD_W - x;
    if (y + h > LCD_H) h = LCD_H - y;
    if (w <= 0 || h <= 0) return;
    ili9341_set_window(x, y, x + w - 1, y + h - 1);
    ili9341_write_color(color, (uint32_t)w * h);
}

void gfx_fill_circle(int16_t cx, int16_t cy, int16_t r, uint16_t color)
{
    for (int16_t dy = -r; dy <= r; dy++) {
        int16_t dx = (int16_t)sqrtf((float)(r*r - dy*dy));
        draw_hline(cx - dx, cy + dy, 2 * dx + 1, color);
    }
}

void gfx_fill_ellipse(int16_t cx, int16_t cy, int16_t rx, int16_t ry, uint16_t color)
{
    if (rx <= 0 || ry <= 0) return;
    for (int16_t dy = -ry; dy <= ry; dy++) {
        float ratio = 1.0f - ((float)dy * dy) / ((float)ry * ry);
        if (ratio < 0) continue;
        int16_t dx = (int16_t)(rx * sqrtf(ratio));
        draw_hline(cx - dx, cy + dy, 2 * dx + 1, color);
    }
}

void gfx_draw_arc(int16_t cx, int16_t cy, int16_t r,
                  int16_t start_deg, int16_t end_deg,
                  int16_t thickness, uint16_t color)
{
    float s = start_deg * M_PI / 180.0f;
    float e = end_deg   * M_PI / 180.0f;
    if (e < s) e += 2 * M_PI;
    int steps = (int)((e - s) * r);
    if (steps < 4) steps = 4;
    for (int i = 0; i <= steps; i++) {
        float angle = s + (e - s) * i / steps;
        for (int16_t t = 0; t < thickness; t++) {
            int16_t rr = r - thickness / 2 + t;
            int16_t px = cx + (int16_t)(rr * cosf(angle));
            int16_t py = cy + (int16_t)(rr * sinf(angle));
            gfx_fill_rect(px - 1, py - 1, 3, 3, color);
        }
    }
}

void gfx_draw_line(int16_t x0, int16_t y0, int16_t x1, int16_t y1,
                   int16_t thickness, uint16_t color)
{
    int16_t dx = abs(x1 - x0), dy = abs(y1 - y0);
    int16_t sx = x0 < x1 ? 1 : -1;
    int16_t sy = y0 < y1 ? 1 : -1;
    int16_t err = dx - dy;
    int16_t half = thickness / 2;
    while (1) {
        gfx_fill_rect(x0 - half, y0 - half, thickness, thickness, color);
        if (x0 == x1 && y0 == y1) break;
        int16_t e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x0 += sx; }
        if (e2 <  dx) { err += dx; y0 += sy; }
    }
}

void gfx_fill_rounded_rect(int16_t x, int16_t y, int16_t w, int16_t h,
                            int16_t r, uint16_t color)
{
    gfx_fill_rect(x + r, y,     w - 2*r, h,     color);
    gfx_fill_rect(x,     y + r, r,       h-2*r, color);
    gfx_fill_rect(x+w-r, y + r, r,       h-2*r, color);
    gfx_fill_circle(x + r,     y + r,     r, color);
    gfx_fill_circle(x + w-r-1, y + r,     r, color);
    gfx_fill_circle(x + r,     y + h-r-1, r, color);
    gfx_fill_circle(x + w-r-1, y + h-r-1, r, color);
}
