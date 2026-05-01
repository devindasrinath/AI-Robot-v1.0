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

void face_set(FaceExpression expr);
void face_tick(void);
void face_init(void);
FaceExpression face_from_string(const char *name);
