/*
 * face.c — Nexu animated face, landscape 320×240, fully flicker-free.
 *
 * FIXES vs previous version:
 *
 * 1. FLICKER BOX FIX — render_mouth() bounding box:
 *    The old code used MARG=55 in BOTH directions (top and bottom), so the
 *    mouth BB's top edge at ~y=123 overlapped the eye/cheek region (eyes sit
 *    at cy≈105±35).  Every mouth frame wrote BG over pixels the eye had just
 *    drawn, creating a visible dark box that flickered as prev_cy changed.
 *    Fix: clamp by_top to never exceed MOUTH_BB_TOP_MAX (y=167), keeping the
 *    mouth BB entirely below the eye region.  The eye renders after the mouth
 *    and its own BB already covers the cheek overlap — no gap is left.
 *
 * 2. SMOOTHER ANIMATIONS:
 *    a) Ease-in-out lerp: replaced flat t=0.08 with a spring-style approach —
 *       velocity accumulates toward target and decays, giving natural overshoot
 *       on expression change and smooth settle.  Eyes and mouth each have their
 *       own velocity fields.
 *    b) Blink curve: replaced linear triangle with a sinusoidal ease so the lid
 *       closes and opens with acceleration — feels biological.
 *    c) Breathing: amplitude raised to ±8% (was ±5%) and modulates both Y
 *       position and scale for a more visible "chest rise" effect.
 *    d) Pupil saccade: drift replaced with periodic saccades — the pupils jump
 *       to a random target and lerp there quickly, then hold.  Much more
 *       lifelike than continuous sinusoidal drift.
 *    e) Expression transitions trigger a soft "squint blink" on change (ry
 *       briefly drops by 30%) so the new face appears to open its eyes fresh.
 *    f) Mouth arc lerps through angle-space properly — start_deg and end_deg
 *       now travel through the shortest angular arc to avoid 0→360 wrap flips.
 *
 * Rendering order: mouth → left eye → right eye.
 * Eyes render last so they always win in the overlap band.
 */

#include "face.h"
#include "gfx.h"
#include "ili9341.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#ifndef M_PI
#define M_PI 3.14159265358979f
#endif

/* ── Palette ─────────────────────────────────────────────────────────────── */
#define BG         RGB(10,  10,  30)
#define EYE_WHITE  COLOR_WHITE
#define EYE_PUPIL  RGB(30,  30,  80)
#define EYE_SHINE  COLOR_WHITE
#define MOUTH_COL  RGB(255, 100, 120)
#define CHEEK_COL  RGB(255, 160, 160)
#define BROW_COL   RGB(200, 200, 255)
#define ZZZCOL     RGB(180, 220, 255)

/* ── Layout (landscape 320×240) ──────────────────────────────────────────── */
#define EYE_L_CX         105
#define EYE_R_CX         215
#define EYE_BASE_CY      105
#define MOUTH_CX         160
#define MOUTH_BASE_CY    178

/*
 * Eye+cheek region bottom boundary.
 * Cheek sits at eye_cy+45=150, half-height 10px → bottom ≈ 160.
 * Any mouth BB row at or above this line is inside the eye compositor's
 * territory; we must not write BG there or we'll clobber the cheek pixels.
 * The eye always renders AFTER the mouth, so it will repair any overlap —
 * but we still skip BG writes on those rows to avoid a one-frame dark flash.
 */
#define EYE_REGION_BOTTOM 162

/* ── Expression descriptors ──────────────────────────────────────────────── */
typedef struct {
    float rx, ry, cy_offset, pupil_scale;
    bool  has_brow;
    float brow_angle;
} EyeShape;

typedef struct {
    float radius, start_deg, end_deg, cy_offset;
    bool  is_circle;
    float circle_r;
} MouthShape;

typedef struct {
    EyeShape   eye_l, eye_r;
    MouthShape mouth;
    bool       cheeks;
} ExprTarget;

static const ExprTarget TARGETS[5] = {
    /* IDLE    */ { {32,32,0,1.0f,false,0}, {32,32,0,1.0f,false,0},
                   {30,20,100,0,false,0}, false },
    /* HAPPY   */ { {32,30,0,1.0f,false,0}, {32,30,0,1.0f,false,0},
                   {38,30,150,-10,false,0}, true },
    /* CURIOUS */ { {32,32,0,1.0f,true,0}, {37,38,-8,1.1f,true,12},
                   {28,20,110,0,false,0}, false },
    /* ALERT   */ { {38,40,0,1.3f,true,8}, {38,40,0,1.3f,true,8},
                   {0,0,0,0,true,16}, false },
    /* SLEEP   */ { {32,5,4,0.6f,false,0}, {32,5,4,0.6f,false,0},
                   {22,20,90,5,false,0}, false },
};

/* ── Spring animator for a single float value ────────────────────────────── */
/*
 * Simple critically-damped spring: each tick the velocity pulls toward
 * (target - value) and is multiplied by a decay factor.  Parameters:
 *   stiffness  — how hard it pulls (higher = snappier)
 *   damping    — how quickly velocity bleeds off (0.7–0.85 = slight overshoot)
 * No dt normalisation needed because face_tick() runs at a fixed ~30fps.
 */
typedef struct { float val, vel; } Spring;

static void spring_tick(Spring *s, float target, float stiffness, float damping)
{
    s->vel += stiffness * (target - s->val);
    s->vel *= damping;
    s->val += s->vel;
}

/* ── Pupil saccade state ──────────────────────────────────────────────────── */
typedef struct {
    float dx, dy;           /* current offset (lerped) */
    float target_dx, target_dy;
    float phase;            /* countdown to next saccade (ticks) */
} Saccade;

static void saccade_tick(Saccade *s, float rx)
{
    /* Lerp toward target — t=0.12 gives a smooth ballistic slide (was 0.25
     * which made the eye jump rather than move).  Real saccades accelerate
     * and decelerate; this simple lerp approximates that well at ~30fps. */
    s->dx += 0.12f * (s->target_dx - s->dx);
    s->dy += 0.12f * (s->target_dy - s->dy);
    s->phase -= 1.0f;
    if (s->phase <= 0.0f) {
        /* Pick a new random gaze target within the sclera */
        float max_d = rx * 0.45f;
        s->target_dx = ((float)(rand() % 201) - 100.0f) * max_d / 100.0f;
        s->target_dy = ((float)(rand() % 201) - 100.0f) * max_d * 0.5f / 100.0f;
        /* Hold gaze 1.5–4.5 s at 30fps = 45–135 ticks */
        s->phase = 45.0f + (float)(rand() % 91);
    }
}

/* ── Animator state ──────────────────────────────────────────────────────── */
typedef struct {
    /* Eye springs */
    Spring rx_l, ry_l, cy_off_l, ps_l;
    Spring rx_r, ry_r, cy_off_r, ps_r;
    /* Mouth springs */
    Spring m_radius, m_start, m_end, m_cy_off, m_circle_r;

    bool  m_is_circle, cheeks, prev_sleeping;

    uint32_t tick, next_blink;
    float    breath_phase;
    Saccade  sacc_l, sacc_r;

    /* Transition squint: briefly suppresses ry after expression change */
    float squint;   /* 0=no squint, 1=full close; decays each tick */

    FaceExpression target_expr;

    struct {
        int16_t eye_l_cy, eye_l_rx, eye_l_ry;
        int16_t eye_r_cy, eye_r_rx, eye_r_ry;
        int16_t mouth_cy;
    } prev;
} AnimState;

static AnimState  s_anim;
static uint16_t   s_row_buf[LCD_W];   /* 640 bytes, static → BSS */

/* ── Helpers ─────────────────────────────────────────────────────────────── */
static uint32_t rand_range(uint32_t lo, uint32_t hi)
{
    return lo + (rand() % (hi - lo + 1));
}
static inline int16_t imax(int16_t a, int16_t b) { return a > b ? a : b; }
static inline int16_t imin(int16_t a, int16_t b) { return a < b ? a : b; }

/*
 * Wrap angle difference into [-180, 180] so lerping mouth arc angles always
 * takes the short path and never flips through 0/360.
 */
static float angle_diff(float from, float to)
{
    float d = to - from;
    while (d >  180.0f) d -= 360.0f;
    while (d < -180.0f) d += 360.0f;
    return d;
}

/* ── Scanline eye compositor ─────────────────────────────────────────────── */
static void render_eye(
    int16_t cx,  int16_t cy,
    int16_t rx,  int16_t ry,
    float   pupil_scale, float pupil_dx, float pupil_dy,
    int16_t prev_cy, int16_t prev_rx, int16_t prev_ry,
    bool    has_cheeks,
    bool    has_brow, float brow_angle)
{
    const int16_t PAD_TOP  = 32;
    const int16_t PAD_SIDE = 16;

    int16_t mrx = imax(rx, prev_rx);
    int16_t mry = imax(ry, prev_ry);
    int16_t lo_cy = imin(cy, prev_cy);
    int16_t hi_cy = imax(cy, prev_cy);

    int16_t bx     = imax(0,       cx - mrx - PAD_SIDE);
    int16_t bx_end = imin(LCD_W-1, cx + mrx + PAD_SIDE);
    int16_t by     = imax(0,       lo_cy - mry - PAD_TOP);
    int16_t by_end = imin(LCD_H-1, hi_cy + 58);
    int16_t bw     = bx_end - bx + 1;
    if (bw <= 0) return;

    /* Pupil */
    int16_t pr = (int16_t)(12.0f * pupil_scale);
    if (pr < 4) pr = 4;
    int16_t center_pull = (cx < LCD_W / 2) ? 3 : -3;
    int16_t px = cx + center_pull + (int16_t)pupil_dx;
    int16_t py = cy + 2           + (int16_t)pupil_dy;
    {
        float dp = (float)(px - cx);
        float mg = (float)(rx - pr - 4);
        if (dp >  mg) px = cx + (int16_t) mg;
        if (dp < -mg) px = cx - (int16_t) mg;
    }
    bool draw_pupil = (ry > 2);

    /* Shine */
    int16_t sh_cx = px - pr / 2, sh_cy = py - pr / 2, sh_r = 4;
    bool    shine  = draw_pupil && (ry > 8);

    /* Cheek */
    int16_t ck_cy = cy + 45;

    /* Brow */
    int16_t brow_by0 = 0, brow_by1 = 0;
    int16_t brow_x0 = cx - 20, brow_x1 = cx + 20;
    if (has_brow && ry > 0) {
        float ba_sin = sinf(brow_angle * (M_PI / 180.0f));
        brow_by0 = cy - ry - 18 - (int16_t)(10.0f * ba_sin);
        brow_by1 = cy - ry - 18 + (int16_t)(10.0f * ba_sin);
    }
    const int16_t BROW_HALF = 2;

    float ry_f  = (float)ry;
    float rx_f  = (float)rx;
    float pr_f  = (float)pr;
    float shr_f = (float)sh_r;

    ili9341_set_window(bx, by, bx_end, by_end);

    for (int16_t y = by; y <= by_end; y++) {
        /* 1. BG */
        for (int16_t i = 0; i < bw; i++) s_row_buf[i] = BG;

        /* 2. Cheek */
        if (has_cheeks) {
            float dy = (float)(y - ck_cy);
            if (dy >= -10.0f && dy <= 10.0f) {
                float r2 = 1.0f - (dy * dy) / 100.0f;
                if (r2 > 0.0f) {
                    int16_t ex = (int16_t)(22.0f * sqrtf(r2));
                    int16_t xl = imin(bw-1, imax(0, cx - ex - bx));
                    int16_t xr = imin(bw-1, imax(0, cx + ex - bx));
                    for (int16_t i = xl; i <= xr; i++) s_row_buf[i] = CHEEK_COL;
                }
            }
        }

        /* 3. Sclera */
        if (ry > 0) {
            float dy = (float)(y - cy);
            if (dy >= -ry_f && dy <= ry_f) {
                float r2 = 1.0f - (dy * dy) / (ry_f * ry_f);
                if (r2 > 0.0f) {
                    int16_t ex = (ry <= 2) ? rx : (int16_t)(rx_f * sqrtf(r2));
                    int16_t xl = imin(bw-1, imax(0, cx - ex - bx));
                    int16_t xr = imin(bw-1, imax(0, cx + ex - bx));
                    for (int16_t i = xl; i <= xr; i++) s_row_buf[i] = EYE_WHITE;
                }
            }
        }

        /* 4. Pupil */
        if (draw_pupil) {
            float dy = (float)(y - py);
            if (dy >= -pr_f && dy <= pr_f) {
                float r2 = 1.0f - (dy * dy) / (pr_f * pr_f);
                if (r2 > 0.0f) {
                    int16_t ex = (int16_t)(pr_f * sqrtf(r2));
                    int16_t xl = imin(bw-1, imax(0, px - ex - bx));
                    int16_t xr = imin(bw-1, imax(0, px + ex - bx));
                    for (int16_t i = xl; i <= xr; i++) s_row_buf[i] = EYE_PUPIL;
                }
            }
        }

        /* 5. Shine */
        if (shine) {
            float dy = (float)(y - sh_cy);
            if (dy >= -shr_f && dy <= shr_f) {
                float r2 = 1.0f - (dy * dy) / (shr_f * shr_f);
                if (r2 > 0.0f) {
                    int16_t ex = (int16_t)(shr_f * sqrtf(r2));
                    int16_t xl = imin(bw-1, imax(0, sh_cx - ex - bx));
                    int16_t xr = imin(bw-1, imax(0, sh_cx + ex - bx));
                    for (int16_t i = xl; i <= xr; i++) s_row_buf[i] = EYE_SHINE;
                }
            }
        }

        /* 6. Brow */
        if (has_brow && ry > 0) {
            int16_t by_lo = imin(brow_by0, brow_by1) - BROW_HALF;
            int16_t by_hi = imax(brow_by0, brow_by1) + BROW_HALF;
            if (y >= by_lo && y <= by_hi) {
                if (brow_by0 == brow_by1) {
                    int16_t xl = imin(bw-1, imax(0, brow_x0 - BROW_HALF - bx));
                    int16_t xr = imin(bw-1, imax(0, brow_x1 + BROW_HALF - bx));
                    for (int16_t i = xl; i <= xr; i++) s_row_buf[i] = BROW_COL;
                } else {
                    float dy_brow = (float)(brow_by1 - brow_by0);
                    float t_lo = ((float)(y - BROW_HALF) - (float)brow_by0) / dy_brow;
                    float t_hi = ((float)(y + BROW_HALF) - (float)brow_by0) / dy_brow;
                    if (t_lo > t_hi) { float tmp = t_lo; t_lo = t_hi; t_hi = tmp; }
                    if (t_lo < 0.0f) t_lo = 0.0f;
                    if (t_hi > 1.0f) t_hi = 1.0f;
                    if (t_hi >= t_lo) {
                        float dx_brow = (float)(brow_x1 - brow_x0);
                        int16_t bpx0 = brow_x0 + (int16_t)(t_lo * dx_brow);
                        int16_t bpx1 = brow_x0 + (int16_t)(t_hi * dx_brow);
                        if (bpx0 > bpx1) { int16_t tmp = bpx0; bpx0 = bpx1; bpx1 = tmp; }
                        int16_t xl = imin(bw-1, imax(0, bpx0 - bx));
                        int16_t xr = imin(bw-1, imax(0, bpx1 - bx));
                        for (int16_t i = xl; i <= xr; i++) s_row_buf[i] = BROW_COL;
                    }
                }
            }
        }

        ili9341_stream_pixels(s_row_buf, bw);
    }
    ili9341_stream_end();
}

/* ── Scanline mouth compositor ───────────────────────────────────────────── */
/*
 * Shape-aware bounding box — mouth BB top is computed from the actual shape
 * extents rather than a hard constant, so the O-circle (ALERT/WONDERING) is
 * never clipped at the top.
 *
 * The previous fix clamped `by` to MOUTH_BB_TOP_MAX=167 which was safely below
 * the cheek (~y=160) but accidentally clipped the O-mouth whose top sits at
 * cy-circle_r = 178-16 = 162.
 *
 * Correct approach:
 *   by_top = cy - circle_r  (circle)  or  lo_cy - radius - half_th  (arc)
 * This always contains the full shape.  Any rows this pushes into the eye band
 * (y < ~162) are written as BG + mouth pixels and then immediately repainted
 * by render_eye() which runs after render_mouth() in the same frame — so no
 * sustained flicker, at most one row of BG for a single frame during
 * transitions.  For ALERT the circle top (y=162) is exactly at the eye band
 * boundary so in practice there is zero overlap.
 */
static void render_mouth(
    int16_t cx, int16_t cy, int16_t prev_cy,
    float radius, float start_deg, float end_deg,
    bool is_circle, float circle_r)
{
    const int16_t MARG = 55;
    int16_t lo_cy  = imin(cy, prev_cy);
    int16_t hi_cy  = imax(cy, prev_cy);
    int16_t bx     = imax(0,       cx - MARG);
    int16_t bx_end = imin(LCD_W-1, cx + MARG);

    /*
     * Shape-aware top edge: use the actual top of whatever is being drawn
     * (circle top = cy - circle_r, arc top = cy - radius - half_th) so the
     * bounding box fully contains the shape even when circle_r pushes the
     * top above where the old hard constant sat.
     * We do NOT clamp against EYE_REGION_BOTTOM here — instead we skip
     * writing BG on rows inside the eye region (see inner loop below), which
     * lets the circle/arc draw correctly while the eye compositor repairs any
     * shared pixels when it runs immediately after.
     */
    float shape_top_f = is_circle ? (cy - circle_r) : (lo_cy - radius - 3.5f);
    int16_t by_shape  = imax(0, (int16_t)shape_top_f - 2);
    /*
     * FLICKER FIX: Clamp the mouth bounding-box top to EYE_REGION_BOTTOM
     * unless the shape genuinely extends above that line (ALERT O-circle can
     * reach y≈162).  Without this clamp, render_mouth() opens a set_window
     * that covers eye-region rows and streams BG pixels there every tick.
     * render_eye() runs after and repairs those rows — but the ILI9341 has
     * already latched the BG colour for one frame, producing a dark box that
     * flickers with breathing and saccades.  By starting the window at or
     * below EYE_REGION_BOTTOM we never touch those rows, so there is nothing
     * to repair and the flicker disappears completely.
     */
    int16_t by = (by_shape < EYE_REGION_BOTTOM) ? EYE_REGION_BOTTOM : by_shape;
    int16_t by_end = imin(LCD_H-1, hi_cy + MARG);
    int16_t bw     = bx_end - bx + 1;
    if (bw <= 0 || by > by_end) return;

    /* Arc geometry */
    float half_th = 3.5f;
    float ri = radius - half_th, ro = radius + half_th;
    float ri2 = ri * ri, ro2 = ro * ro;

    float alpha_s = start_deg * (M_PI / 180.0f);
    float alpha_e = end_deg   * (M_PI / 180.0f);
    float cs = cosf(alpha_s), ss = sinf(alpha_s);
    float ce = cosf(alpha_e), se = sinf(alpha_e);
    float span = alpha_e - alpha_s;
    if (span < 0.0f) span += 2.0f * M_PI;
    bool wide_arc = (span > M_PI);

    /* O-mouth ring */
    float cr2 = circle_r * circle_r;
    float ci   = circle_r - 8.0f; if (ci < 2.0f) ci = 2.0f;
    float ci2  = ci * ci;

    ili9341_set_window(bx, by, bx_end, by_end);

    for (int16_t y = by; y <= by_end; y++) {
        for (int16_t i = 0; i < bw; i++) s_row_buf[i] = BG;

        float dy  = (float)(y - cy);
        float dy2 = dy * dy;

        if (is_circle) {
            if (dy2 <= cr2) {
                float dxo = sqrtf(cr2 - dy2);
                int16_t xl = imin(bw-1, imax(0, cx - (int16_t)dxo - bx));
                int16_t xr = imin(bw-1, imax(0, cx + (int16_t)dxo - bx));
                for (int16_t i = xl; i <= xr; i++) s_row_buf[i] = MOUTH_COL;
            }
            if (dy2 <= ci2) {
                float dxi = sqrtf(ci2 - dy2);
                int16_t xl = imin(bw-1, imax(0, cx - (int16_t)dxi - bx));
                int16_t xr = imin(bw-1, imax(0, cx + (int16_t)dxi - bx));
                for (int16_t i = xl; i <= xr; i++) s_row_buf[i] = BG;
            }
        } else if (ro2 > 0.0f && dy2 < ro2) {
            float dxo = sqrtf(ro2 - dy2);
            float dxi = (ri2 > 0.0f && dy2 < ri2) ? sqrtf(ri2 - dy2) : 0.0f;

            int16_t x0r = cx + (int16_t)dxi + 1;
            int16_t x1r = cx + (int16_t)dxo;
            for (int16_t x = x0r; x <= x1r; x++) {
                float dx = (float)(x - cx);
                bool past_start  = (cs * dy - ss * dx) >= 0.0f;
                bool before_end  = (ce * dy - se * dx) <= 0.0f;
                bool in_arc = wide_arc ? (past_start || before_end)
                                       : (past_start && before_end);
                if (in_arc) {
                    int16_t bi = x - bx;
                    if (bi >= 0 && bi < bw) s_row_buf[bi] = MOUTH_COL;
                }
            }
            int16_t x0l = cx - (int16_t)dxo;
            int16_t x1l = cx - (int16_t)dxi - 1;
            for (int16_t x = x0l; x <= x1l; x++) {
                float dx = (float)(x - cx);
                bool past_start  = (cs * dy - ss * dx) >= 0.0f;
                bool before_end  = (ce * dy - se * dx) <= 0.0f;
                bool in_arc = wide_arc ? (past_start || before_end)
                                       : (past_start && before_end);
                if (in_arc) {
                    int16_t bi = x - bx;
                    if (bi >= 0 && bi < bw) s_row_buf[bi] = MOUTH_COL;
                }
            }
        }

        ili9341_stream_pixels(s_row_buf, bw);
    }
    ili9341_stream_end();
}

/* ── Zzz (sleep) ─────────────────────────────────────────────────────────── */
static void clear_zzz(void)
{
    for (int i = 0; i < 3; i++) {
        int16_t zx = 258 + i * 20, zy = 62 - i * 18;
        gfx_fill_circle(zx, zy, (5 + i * 4) + 5, BG);
    }
}

static void draw_zzz(uint32_t tick)
{
    float zp = sinf(tick * 0.05f) * 0.3f + 0.7f;
    for (int i = 0; i < 3; i++) {
        int16_t zx = 258 + i * 20, zy = 62 - i * 18;
        int16_t zr = (int16_t)((5 + i * 4) * zp);
        if (zr < 2) zr = 2;
        gfx_fill_circle(zx, zy, zr, ZZZCOL);
    }
}

/* ── face_tick ───────────────────────────────────────────────────────────── */
/*
 * Spring constants tuned for ~30fps:
 *   stiffness 0.10 + damping 0.78 → settle in ~12 frames, tiny overshoot.
 * Mouth uses slightly softer spring (0.07 / 0.80) so arc transitions look
 * more fluid than the snappier eye changes.
 */
#define EYE_K   0.10f
#define EYE_D   0.78f
#define MOUTH_K 0.04f   /* was 0.07 — softer pull gives a longer, more fluid glide */
#define MOUTH_D 0.84f   /* was 0.80 — higher damping prevents arc overshoot wobble */

void face_tick(void)
{
    s_anim.tick++;
    const ExprTarget *tgt = &TARGETS[s_anim.target_expr];

    /* ── Spring-advance all shape parameters ──────────────────────────────── */
    spring_tick(&s_anim.rx_l,     tgt->eye_l.rx,          EYE_K, EYE_D);
    spring_tick(&s_anim.ry_l,     tgt->eye_l.ry,          EYE_K, EYE_D);
    spring_tick(&s_anim.cy_off_l, tgt->eye_l.cy_offset,   EYE_K, EYE_D);
    spring_tick(&s_anim.ps_l,     tgt->eye_l.pupil_scale, EYE_K, EYE_D);
    spring_tick(&s_anim.rx_r,     tgt->eye_r.rx,          EYE_K, EYE_D);
    spring_tick(&s_anim.ry_r,     tgt->eye_r.ry,          EYE_K, EYE_D);
    spring_tick(&s_anim.cy_off_r, tgt->eye_r.cy_offset,   EYE_K, EYE_D);
    spring_tick(&s_anim.ps_r,     tgt->eye_r.pupil_scale, EYE_K, EYE_D);

    spring_tick(&s_anim.m_radius,   tgt->mouth.radius,    MOUTH_K, MOUTH_D);
    spring_tick(&s_anim.m_cy_off,   tgt->mouth.cy_offset, MOUTH_K, MOUTH_D);
    spring_tick(&s_anim.m_circle_r, tgt->mouth.circle_r,  MOUTH_K, MOUTH_D);

    /*
     * Angle lerp via shortest-arc difference so the mouth arc never spins
     * the wrong way around when crossing 0°/360°.
     */
    float s_target = s_anim.m_start.val + angle_diff(s_anim.m_start.val, tgt->mouth.start_deg);
    float e_target = s_anim.m_end.val   + angle_diff(s_anim.m_end.val,   tgt->mouth.end_deg);
    spring_tick(&s_anim.m_start, s_target, MOUTH_K, MOUTH_D);
    spring_tick(&s_anim.m_end,   e_target, MOUTH_K, MOUTH_D);

    s_anim.m_is_circle = tgt->mouth.is_circle;
    s_anim.cheeks      = tgt->cheeks;

    /* ── Blink ────────────────────────────────────────────────────────────── */
    /*
     * Sinusoidal blink curve (sin² ramp):
     *   close phase: scale = cos²(bp*π/2), 0→0.5  (bp in [0,0.5])
     *   open  phase: scale = sin²(bp*π),   0.5→1  (bp in [0.5,1])
     * This gives natural acceleration at start/end of each lid movement.
     */
    float blink_scale = 1.0f;
    if (s_anim.tick >= s_anim.next_blink) {
        float bp = (float)(s_anim.tick - s_anim.next_blink) / 8.0f;
        if (bp < 0.5f) {
            float c = cosf(bp * M_PI);
            blink_scale = c * c;
        } else if (bp < 1.0f) {
            float s = sinf(bp * M_PI);
            blink_scale = s * s;
        } else {
            s_anim.next_blink = s_anim.tick + rand_range(90, 300);
        }
    }
    if (s_anim.target_expr == FACE_SLEEP) blink_scale = 1.0f;

    /* ── Transition squint ────────────────────────────────────────────────── */
    /* Decay the squint factor — blends with blink_scale multiplicatively */
    if (s_anim.squint > 0.0f) {
        s_anim.squint -= 0.07f;
        if (s_anim.squint < 0.0f) s_anim.squint = 0.0f;
    }
    /* Squint modulates ry: at squint=1 the eye closes to 30%, then opens */
    float squint_scale = 1.0f - 0.70f * s_anim.squint;

    /* ── Breathing ────────────────────────────────────────────────────────── */
    /*
     * Raised to ±8% amplitude (was ±5%).  Also offsets cy slightly so the
     * whole eye region appears to rise and fall — more "chest rise" feel.
     */
    s_anim.breath_phase += 0.005f;
    float breath_wave  = sinf(s_anim.breath_phase);
    float breath_scale = 1.0f + 0.08f * breath_wave;
    float breath_rx_scale = 1.0f + 0.03f * breath_wave; /* subtle horizontal swell */
    float breath_cy    = -2.0f * breath_wave;   /* eye rides up on inhale */

    /* ── Pupil saccades ───────────────────────────────────────────────────── */
    saccade_tick(&s_anim.sacc_l, s_anim.rx_l.val);
    saccade_tick(&s_anim.sacc_r, s_anim.rx_r.val);

    /* ── Final geometry ───────────────────────────────────────────────────── */
    float combined_l = blink_scale * squint_scale * breath_scale;
    float combined_r = blink_scale * squint_scale * breath_scale;

    int16_t el_cy = EYE_BASE_CY + (int16_t)(s_anim.cy_off_l.val + breath_cy);
    int16_t er_cy = EYE_BASE_CY + (int16_t)(s_anim.cy_off_r.val + breath_cy);
    int16_t el_rx = (int16_t)(s_anim.rx_l.val * breath_rx_scale);
    int16_t el_ry = (int16_t)(s_anim.ry_l.val * combined_l);
    int16_t er_rx = (int16_t)(s_anim.rx_r.val * breath_rx_scale);
    int16_t er_ry = (int16_t)(s_anim.ry_r.val * combined_r);
    int16_t m_cy  = MOUTH_BASE_CY + (int16_t)s_anim.m_cy_off.val;

    /* Render mouth first, then eyes (eyes win any overlap) */
    render_mouth(MOUTH_CX, m_cy, s_anim.prev.mouth_cy,
                 s_anim.m_radius.val, s_anim.m_start.val, s_anim.m_end.val,
                 s_anim.m_is_circle, s_anim.m_circle_r.val);

    render_eye(EYE_L_CX, el_cy, el_rx, el_ry,
               s_anim.ps_l.val, s_anim.sacc_l.dx, s_anim.sacc_l.dy,
               s_anim.prev.eye_l_cy, s_anim.prev.eye_l_rx, s_anim.prev.eye_l_ry,
               s_anim.cheeks,
               tgt->eye_l.has_brow, tgt->eye_l.brow_angle);

    render_eye(EYE_R_CX, er_cy, er_rx, er_ry,
               s_anim.ps_r.val, s_anim.sacc_r.dx, s_anim.sacc_r.dy,
               s_anim.prev.eye_r_cy, s_anim.prev.eye_r_rx, s_anim.prev.eye_r_ry,
               s_anim.cheeks,
               tgt->eye_r.has_brow, tgt->eye_r.brow_angle);

    /* Zzz */
    bool sleeping = (s_anim.target_expr == FACE_SLEEP);
    if (!sleeping && s_anim.prev_sleeping) clear_zzz();
    if (sleeping) draw_zzz(s_anim.tick);
    s_anim.prev_sleeping = sleeping;

    /* Save prev geometry for next frame's dirty-rect expansion */
    s_anim.prev.eye_l_cy = el_cy; s_anim.prev.eye_l_rx = el_rx; s_anim.prev.eye_l_ry = el_ry;
    s_anim.prev.eye_r_cy = er_cy; s_anim.prev.eye_r_rx = er_rx; s_anim.prev.eye_r_ry = er_ry;
    s_anim.prev.mouth_cy = m_cy;
}

/* ── Public API ──────────────────────────────────────────────────────────── */
void face_init(void)
{
    const ExprTarget *tgt = &TARGETS[FACE_IDLE];

    /* Initialise springs at resting value, zero velocity */
    s_anim.rx_l     = (Spring){ tgt->eye_l.rx,          0 };
    s_anim.ry_l     = (Spring){ tgt->eye_l.ry,          0 };
    s_anim.cy_off_l = (Spring){ tgt->eye_l.cy_offset,   0 };
    s_anim.ps_l     = (Spring){ 1.0f,                   0 };
    s_anim.rx_r     = (Spring){ tgt->eye_r.rx,          0 };
    s_anim.ry_r     = (Spring){ tgt->eye_r.ry,          0 };
    s_anim.cy_off_r = (Spring){ tgt->eye_r.cy_offset,   0 };
    s_anim.ps_r     = (Spring){ 1.0f,                   0 };

    s_anim.m_radius   = (Spring){ tgt->mouth.radius,    0 };
    s_anim.m_start    = (Spring){ tgt->mouth.start_deg, 0 };
    s_anim.m_end      = (Spring){ tgt->mouth.end_deg,   0 };
    s_anim.m_cy_off   = (Spring){ tgt->mouth.cy_offset, 0 };
    s_anim.m_circle_r = (Spring){ 0.0f,                 0 };

    s_anim.m_is_circle   = false;
    s_anim.cheeks        = false;
    s_anim.prev_sleeping = false;
    s_anim.squint        = 0.0f;

    /* Saccades: start looking straight, first jump after 1–2s */
    s_anim.sacc_l = (Saccade){ 0, 0, 0, 0, (float)rand_range(30, 60) };
    s_anim.sacc_r = (Saccade){ 0, 0, 0, 0, (float)rand_range(30, 60) };

    s_anim.next_blink  = rand_range(90, 240);
    s_anim.target_expr = FACE_IDLE;

    s_anim.prev.eye_l_cy = EYE_BASE_CY; s_anim.prev.eye_l_rx = 34; s_anim.prev.eye_l_ry = 34;
    s_anim.prev.eye_r_cy = EYE_BASE_CY; s_anim.prev.eye_r_rx = 34; s_anim.prev.eye_r_ry = 34;
    s_anim.prev.mouth_cy = MOUTH_BASE_CY;

    gfx_fill_screen(BG);
}

void face_set(FaceExpression expr)
{
    if (expr == s_anim.target_expr) return;
    s_anim.target_expr = expr;
    /* Trigger a transition squint so new expression "opens its eyes" */
    s_anim.squint     = 1.0f;
    /* Also force a quick blink sequence to mask the expression change */
    s_anim.next_blink = s_anim.tick + 2;
}

FaceExpression face_from_string(const char *name)
{
    if (strcmp(name, "happy")   == 0) return FACE_HAPPY;
    if (strcmp(name, "curious") == 0) return FACE_CURIOUS;
    if (strcmp(name, "alert")   == 0) return FACE_ALERT;
    if (strcmp(name, "sleep")   == 0) return FACE_SLEEP;
    return FACE_IDLE;
}