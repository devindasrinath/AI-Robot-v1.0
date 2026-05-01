#pragma once
#include <stdint.h>
#include <stdbool.h>

typedef enum {
    FACE_IDLE    = 0,
    FACE_HAPPY   = 1,
    FACE_CURIOUS = 2,
    FACE_ALERT   = 3,
    FACE_SLEEP   = 4,
} FaceExpression;

/*
 * Set target expression. The face animator smoothly transitions
 * from whatever it is currently drawing toward the new expression.
 * Never call face_draw() directly — just set the target.
 */
void face_set(FaceExpression expr);

/*
 * Call every 16ms from display_task.
 * Advances all animations by one frame: transitions, blinks,
 * breathing, pupil drift. Redraws only changed regions.
 */
void face_tick(void);

/* Boot init — call once after st7789_init() */
void face_init(void);

FaceExpression face_from_string(const char *name);
