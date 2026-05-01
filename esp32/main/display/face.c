/*
 * face.c — Nexu animated face renderer (60fps, fluid transitions)
 *
 * Design philosophy: the face is NEVER static. Every frame something moves.
 * Expressions are targets — the animator interpolates smoothly toward them.
 * This is what makes Nexu feel alive, not like a machine showing images.
 *
 * Animation layers (all run simultaneously):
 *  1. Expression morph  — eye/mouth shape lerps toward target over ~300ms
 *  2. Blink             — random interval 2–6s, 120ms duration, smooth close/open
 *  3. Pupil drift       — pupils wander slightly (±6px) with slow perlin-like noise
 *  4. Breathing scale   — eye vertical radius breathes ±2px at 0.15Hz
 *  5. Expression squash — on new expression: brief squash then settle
 *
 * Rendering: only dirty regions are redrawn each frame to hit 60fps.
 */

#include "face.h"
#include "gfx.h"
#include "st7789.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#ifndef M_PI
#define M_PI 3.14159265358979f
#endif

/* ── Palette ─────────────────────────────────────────────────────────────── */
#define BG          RGB(10,  10,  30)
#define EYE_WHITE   COLOR_WHITE
#define EYE_PUPIL   RGB(30,  30,  80)
#define EYE_SHINE   COLOR_WHITE
#define MOUTH_COL   RGB(255, 100, 120)
#define CHEEK_COL   RGB(255, 160, 160)
#define BROW_COL    RGB(200, 200, 255)
#define ZZZCOL      RGB(180, 220, 255)

/* ── Screen layout ───────────────────────────────────────────────────────── */
#define EYE_L_CX    75
#define EYE_R_CX    165
#define EYE_BASE_CY 120
#define MOUTH_CX    120
#define MOUTH_BASE_CY 225

/* ── Eye shape descriptor ────────────────────────────────────────────────── */
typedef struct {
    float rx;          /* horizontal radius */
    float ry;          /* vertical radius (0 = closed line) */
    float cy_offset;   /* vertical offset from base */
    float pupil_scale; /* 1.0 = normal, 1.3 = wide */
    bool  has_brow;
    float brow_angle;  /* degrees: + = raised */
} EyeShape;

/* ── Mouth shape descriptor ──────────────────────────────────────────────── */
typedef struct {
    float radius;
    float start_deg;   /* arc start angle */
    float end_deg;     /* arc end angle   */
    float cy_offset;
    bool  is_circle;   /* true = O mouth (alert) */
    float circle_r;
} MouthShape;

/* ── Expression targets ──────────────────────────────────────────────────── */
typedef struct {
    EyeShape   eye_l, eye_r;
    MouthShape mouth;
    bool       cheeks;
} ExprTarget;

static const ExprTarget TARGETS[5] = {
    /* IDLE */
    {
        .eye_l  = {32, 32,  0, 1.0f, false, 0},
        .eye_r  = {32, 32,  0, 1.0f, false, 0},
        .mouth  = {30, 20, 100, 0, false, 0},
        .cheeks = false,
    },
    /* HAPPY */
    {
        .eye_l  = {32, 30,  0, 1.0f, false, 0},
        .eye_r  = {32, 30,  0, 1.0f, false, 0},
        .mouth  = {38, 30, 150, -10, false, 0},
        .cheeks = true,
    },
    /* CURIOUS */
    {
        .eye_l  = {32, 32,  0,  1.0f, true, 0},
        .eye_r  = {37, 38, -8,  1.1f, true, 12},
        .mouth  = {28, 20, 110, 0, false, 0},
        .cheeks = false,
    },
    /* ALERT */
    {
        .eye_l  = {38, 40,  0,  1.3f, true, 8},
        .eye_r  = {38, 40,  0,  1.3f, true, 8},
        .mouth  = {0,   0,   0,  0, true, 16},
        .cheeks = false,
    },
    /* SLEEP */
    {
        .eye_l  = {32,  5,  4,  0.6f, false, 0},
        .eye_r  = {32,  5,  4,  0.6f, false, 0},
        .mouth  = {22, 20,  90, 5, false, 0},
        .cheeks = false,
    },
};

/* ── Animator state ──────────────────────────────────────────────────────── */

typedef struct {
    /* Current interpolated eye values */
    float rx_l, ry_l, cy_off_l, ps_l;
    float rx_r, ry_r, cy_off_r, ps_r;
    float m_radius, m_start, m_end, m_cy_off;
    float m_circle_r;
    bool  m_is_circle;
    bool  cheeks;

    /* Blink state */
    uint32_t tick;
    uint32_t next_blink;    /* tick when next blink starts */
    float    blink_phase;   /* 0=open, 0.5=closed, 1=open again */
    bool     blinking;

    /* Pupil drift (separate for each eye) */
    float pupil_dx_l, pupil_dy_l;
    float pupil_dx_r, pupil_dy_r;
    float drift_phase_l, drift_phase_r;

    /* Breathing */
    float breath_phase;

    /* Squash on new expression */
    float squash;          /* 1.0=normal, <1=squashed */
    uint32_t squash_tick;

    FaceExpression current_expr;
    FaceExpression target_expr;

    /* Previous frame geometry (for dirty-rect clearing) */
    struct {
        int16_t eye_l_cx, eye_l_cy, eye_l_rx, eye_l_ry;
        int16_t eye_r_cx, eye_r_cy, eye_r_rx, eye_r_ry;
        int16_t mouth_cx, mouth_cy, mouth_r;
    } prev;
} AnimState;

static AnimState s_anim;

/* ── Lerp helpers ────────────────────────────────────────────────────────── */

static float lerpf(float a, float b, float t)
{
    if (t > 1) t = 1;
    return a + (b - a) * t;
}

static uint32_t rand_range(uint32_t lo, uint32_t hi)
{
    return lo + (rand() % (hi - lo + 1));
}

/* ── Draw one eye ────────────────────────────────────────────────────────── */

static void draw_one_eye(int16_t cx, int16_t cy,
                          int16_t rx, int16_t ry,
                          float pupil_scale,
                          float pupil_dx, float pupil_dy)
{
    if (ry <= 2) {
        /* Fully or nearly closed — draw a thin line */
        gfx_fill_ellipse(cx, cy, rx, 3 > ry ? 3 : ry, EYE_WHITE);
        return;
    }

    /* White sclera */
    gfx_fill_ellipse(cx, cy, rx, ry, EYE_WHITE);

    /* Pupil — offset toward center of face + drift */
    int16_t center_pull = (cx < 120) ? 3 : -3;
    int16_t px = cx + center_pull + (int16_t)pupil_dx;
    int16_t py = cy + 2           + (int16_t)pupil_dy;
    int16_t pr = (int16_t)(12 * pupil_scale);

    /* Clamp pupil inside sclera */
    float margin = rx - pr - 4;
    float dx = px - cx;
    if (dx >  margin) px = cx + (int16_t) margin;
    if (dx < -margin) px = cx - (int16_t) margin;

    gfx_fill_circle(px, py, pr, EYE_PUPIL);

    /* Shine dot (top-left of pupil) */
    if (ry > 8)
        gfx_fill_circle(px - pr/2, py - pr/2, 4, EYE_SHINE);
}

/* ── Draw mouth ──────────────────────────────────────────────────────────── */

static void draw_mouth(int16_t cx, int16_t cy,
                        float radius, float start, float end,
                        bool is_circle, float circle_r)
{
    if (is_circle) {
        /* Alert O-mouth */
        int16_t cr = (int16_t)circle_r;
        gfx_fill_circle(cx, cy, cr, MOUTH_COL);
        gfx_fill_circle(cx, cy, cr - 8 > 2 ? cr - 8 : 2, BG);
    } else {
        gfx_draw_arc(cx, cy, (int16_t)radius,
                     (int16_t)start, (int16_t)end, 7, MOUTH_COL);
    }
}

/* ── Clear previous frame regions ────────────────────────────────────────── */

static void clear_eye_region(int16_t cx, int16_t cy, int16_t rx, int16_t ry)
{
    int16_t pad = 14;  /* extra padding to cover pupil + brow */
    gfx_fill_rect(cx - rx - pad, cy - ry - pad,
                  (rx + pad) * 2, (ry + pad) * 2 + 30, BG);
}

static void clear_mouth_region(int16_t cx, int16_t cy, int16_t r)
{
    int16_t pad = 12;
    gfx_fill_rect(cx - r - pad, cy - r - pad,
                  (r + pad) * 2, (r + pad) * 2, BG);
}

/* ── face_tick: advance one frame ────────────────────────────────────────── */

void face_tick(void)
{
    s_anim.tick++;
    float t = 0.08f;  /* lerp speed: ~300ms to full transition at 60fps */

    const ExprTarget *tgt = &TARGETS[s_anim.target_expr];

    /* ── 1. Lerp eye shapes toward target ─────────────────────────────── */
    s_anim.rx_l     = lerpf(s_anim.rx_l,     tgt->eye_l.rx,         t);
    s_anim.ry_l     = lerpf(s_anim.ry_l,     tgt->eye_l.ry,         t);
    s_anim.cy_off_l = lerpf(s_anim.cy_off_l, tgt->eye_l.cy_offset,  t);
    s_anim.ps_l     = lerpf(s_anim.ps_l,     tgt->eye_l.pupil_scale,t);

    s_anim.rx_r     = lerpf(s_anim.rx_r,     tgt->eye_r.rx,         t);
    s_anim.ry_r     = lerpf(s_anim.ry_r,     tgt->eye_r.ry,         t);
    s_anim.cy_off_r = lerpf(s_anim.cy_off_r, tgt->eye_r.cy_offset,  t);
    s_anim.ps_r     = lerpf(s_anim.ps_r,     tgt->eye_r.pupil_scale,t);

    /* ── 2. Lerp mouth toward target ──────────────────────────────────── */
    s_anim.m_radius   = lerpf(s_anim.m_radius,   tgt->mouth.radius,    t);
    s_anim.m_start    = lerpf(s_anim.m_start,     tgt->mouth.start_deg, t);
    s_anim.m_end      = lerpf(s_anim.m_end,       tgt->mouth.end_deg,   t);
    s_anim.m_cy_off   = lerpf(s_anim.m_cy_off,    tgt->mouth.cy_offset, t);
    s_anim.m_circle_r = lerpf(s_anim.m_circle_r,  tgt->mouth.circle_r,  t);
    s_anim.m_is_circle = tgt->mouth.is_circle;
    s_anim.cheeks      = tgt->cheeks;

    /* ── 3. Blink ──────────────────────────────────────────────────────── */
    float blink_ry_scale = 1.0f;
    if (s_anim.tick >= s_anim.next_blink) {
        float elapsed = (float)(s_anim.tick - s_anim.next_blink);
        float blink_ticks = 7.5f;  /* 120ms at 60fps */
        float bp = elapsed / blink_ticks;

        if (bp < 0.5f)
            blink_ry_scale = 1.0f - 2.0f * bp;       /* closing */
        else if (bp < 1.0f)
            blink_ry_scale = 2.0f * (bp - 0.5f);     /* opening */
        else {
            /* Blink complete — schedule next */
            blink_ry_scale = 1.0f;
            s_anim.next_blink = s_anim.tick + rand_range(120, 360); /* 2-6s */
        }
    }

    /* Don't blink in sleep (eyes already near-closed) */
    if (s_anim.target_expr == FACE_SLEEP) blink_ry_scale = 1.0f;

    /* ── 4. Breathing — slow vertical pulse on eye ry ─────────────────── */
    s_anim.breath_phase += 0.006f;  /* ~0.18Hz */
    float breath = 1.0f + 0.05f * sinf(s_anim.breath_phase);

    /* ── 5. Pupil drift — slow wandering (different phase each eye) ────── */
    s_anim.drift_phase_l += 0.018f;
    s_anim.drift_phase_r += 0.022f;
    s_anim.pupil_dx_l = 5.0f * sinf(s_anim.drift_phase_l * 1.3f);
    s_anim.pupil_dy_l = 4.0f * sinf(s_anim.drift_phase_l * 0.9f);
    s_anim.pupil_dx_r = 5.0f * sinf(s_anim.drift_phase_r * 1.1f);
    s_anim.pupil_dy_r = 4.0f * sinf(s_anim.drift_phase_r * 1.4f);

    /* ── 6. Compute final geometry ─────────────────────────────────────── */
    int16_t el_cx = EYE_L_CX;
    int16_t er_cx = EYE_R_CX;
    int16_t el_cy = EYE_BASE_CY + (int16_t)s_anim.cy_off_l;
    int16_t er_cy = EYE_BASE_CY + (int16_t)s_anim.cy_off_r;
    int16_t el_rx = (int16_t)(s_anim.rx_l);
    int16_t er_rx = (int16_t)(s_anim.rx_r);
    int16_t el_ry = (int16_t)(s_anim.ry_l * blink_ry_scale * breath);
    int16_t er_ry = (int16_t)(s_anim.ry_r * blink_ry_scale * breath);
    int16_t m_cy  = MOUTH_BASE_CY + (int16_t)s_anim.m_cy_off;

    /* ── 7. Clear only changed regions ─────────────────────────────────── */
    /* Clear with generous padding to erase previous frame's pupil + drift */
    clear_eye_region(el_cx, s_anim.prev.eye_l_cy,
                     s_anim.prev.eye_l_rx + 6, s_anim.prev.eye_l_ry + 6);
    clear_eye_region(er_cx, s_anim.prev.eye_r_cy,
                     s_anim.prev.eye_r_rx + 6, s_anim.prev.eye_r_ry + 6);
    clear_mouth_region(MOUTH_CX, s_anim.prev.mouth_cy,
                       s_anim.prev.mouth_r + 8);

    /* ── 8. Draw this frame ─────────────────────────────────────────────── */
    /* Cheeks (only in happy — appears/disappears with transition) */
    if (s_anim.cheeks) {
        gfx_fill_ellipse(EYE_L_CX, el_cy + 45, 22, 10, CHEEK_COL);
        gfx_fill_ellipse(EYE_R_CX, er_cy + 45, 22, 10, CHEEK_COL);
    }

    /* Eyes */
    draw_one_eye(el_cx, el_cy, el_rx, el_ry, s_anim.ps_l,
                 s_anim.pupil_dx_l, s_anim.pupil_dy_l);
    draw_one_eye(er_cx, er_cy, er_rx, er_ry, s_anim.ps_r,
                 s_anim.pupil_dx_r, s_anim.pupil_dy_r);

    /* Eyebrows for curious/alert */
    if (tgt->eye_r.has_brow) {
        float ba = tgt->eye_r.brow_angle * M_PI / 180.0f;
        int16_t bx0 = er_cx - 20;
        int16_t bx1 = er_cx + 20;
        int16_t by0 = er_cy - er_ry - 18 - (int16_t)(10 * sinf(ba));
        int16_t by1 = er_cy - er_ry - 18 + (int16_t)(10 * sinf(ba));
        gfx_draw_line(bx0, by0, bx1, by1, 4, BROW_COL);
    }
    if (tgt->eye_l.has_brow) {
        int16_t bx0 = el_cx - 20;
        int16_t bx1 = el_cx + 20;
        int16_t by  = el_cy - el_ry - 18;
        gfx_draw_line(bx0, by, bx1, by, 4, BROW_COL);
    }

    /* Mouth */
    draw_mouth(MOUTH_CX, m_cy,
               s_anim.m_radius, s_anim.m_start, s_anim.m_end,
               s_anim.m_is_circle, s_anim.m_circle_r);

    /* Sleep Zzz dots — slowly pulse size */
    if (s_anim.target_expr == FACE_SLEEP) {
        float zp = sinf(s_anim.tick * 0.05f) * 0.3f + 0.7f;
        for (int i = 0; i < 3; i++) {
            int16_t zx = 158 + i * 22;
            int16_t zy = 82  - i * 22;
            int16_t zr = (int16_t)((5 + i * 4) * zp);
            gfx_fill_circle(zx, zy, zr + 2, BG);  /* erase prev */
            gfx_fill_circle(zx, zy, zr,     ZZZCOL);
        }
    }

    /* ── 9. Save geometry for next frame's clear ────────────────────────── */
    s_anim.prev.eye_l_cx = el_cx;
    s_anim.prev.eye_l_cy = el_cy;
    s_anim.prev.eye_l_rx = el_rx;
    s_anim.prev.eye_l_ry = el_ry;
    s_anim.prev.eye_r_cx = er_cx;
    s_anim.prev.eye_r_cy = er_cy;
    s_anim.prev.eye_r_rx = er_rx;
    s_anim.prev.eye_r_ry = er_ry;
    s_anim.prev.mouth_cy = m_cy;
    s_anim.prev.mouth_r  = (int16_t)(s_anim.m_is_circle ?
                                      s_anim.m_circle_r : s_anim.m_radius);
}

/* ── Public API ──────────────────────────────────────────────────────────── */

void face_init(void)
{
    const ExprTarget *t = &TARGETS[FACE_IDLE];
    s_anim.rx_l = t->eye_l.rx;   s_anim.ry_l = t->eye_l.ry;
    s_anim.rx_r = t->eye_r.rx;   s_anim.ry_r = t->eye_r.ry;
    s_anim.ps_l = 1.0f;          s_anim.ps_r = 1.0f;
    s_anim.m_radius = t->mouth.radius;
    s_anim.m_start  = t->mouth.start_deg;
    s_anim.m_end    = t->mouth.end_deg;
    s_anim.m_circle_r = 0;
    s_anim.next_blink = rand_range(120, 300);
    s_anim.breath_phase = 0;
    s_anim.drift_phase_l = 0;
    s_anim.drift_phase_r = 1.5f;  /* offset so eyes drift differently */
    s_anim.current_expr = FACE_IDLE;
    s_anim.target_expr  = FACE_IDLE;

    /* Initial prev geometry to avoid huge clear on first tick */
    s_anim.prev.eye_l_cx = EYE_L_CX; s_anim.prev.eye_l_cy = EYE_BASE_CY;
    s_anim.prev.eye_l_rx = 34;        s_anim.prev.eye_l_ry = 34;
    s_anim.prev.eye_r_cx = EYE_R_CX; s_anim.prev.eye_r_cy = EYE_BASE_CY;
    s_anim.prev.eye_r_rx = 34;        s_anim.prev.eye_r_ry = 34;
    s_anim.prev.mouth_cy = MOUTH_BASE_CY;
    s_anim.prev.mouth_r  = 40;

    gfx_fill_screen(BG);
}

void face_set(FaceExpression expr)
{
    s_anim.target_expr = expr;
    /* Trigger a brief squash to acknowledge the change */
    s_anim.next_blink = s_anim.tick + 8;  /* blink on expression change */
}

FaceExpression face_from_string(const char *name)
{
    if (strcmp(name, "happy")   == 0) return FACE_HAPPY;
    if (strcmp(name, "curious") == 0) return FACE_CURIOUS;
    if (strcmp(name, "alert")   == 0) return FACE_ALERT;
    if (strcmp(name, "sleep")   == 0) return FACE_SLEEP;
    return FACE_IDLE;
}
