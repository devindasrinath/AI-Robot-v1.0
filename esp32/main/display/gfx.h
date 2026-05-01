#pragma once
#include <stdint.h>
#include "st7789.h"

void gfx_fill_screen(uint16_t color);
void gfx_fill_rect(int16_t x, int16_t y, int16_t w, int16_t h, uint16_t color);
void gfx_fill_circle(int16_t cx, int16_t cy, int16_t r, uint16_t color);
void gfx_fill_ellipse(int16_t cx, int16_t cy, int16_t rx, int16_t ry, uint16_t color);

/* Arc: angles in degrees, 0=right, 90=down, clockwise */
void gfx_draw_arc(int16_t cx, int16_t cy, int16_t r,
                  int16_t start_deg, int16_t end_deg,
                  int16_t thickness, uint16_t color);

void gfx_draw_line(int16_t x0, int16_t y0, int16_t x1, int16_t y1,
                   int16_t thickness, uint16_t color);

void gfx_fill_rounded_rect(int16_t x, int16_t y, int16_t w, int16_t h,
                            int16_t r, uint16_t color);
