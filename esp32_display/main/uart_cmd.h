#pragma once

/* ── Inter-C3 UART (Main C3 → Display C3) ───────────────────────────────────
 *
 *  Protocol: text lines, same format as SG2002 → Main C3:
 *    "DISPLAY:idle\n"
 *    "DISPLAY:happy\n"
 *    "DISPLAY:curious\n"
 *    "DISPLAY:alert\n"
 *    "DISPLAY:sleep\n"
 *
 *  Wiring:
 *    Main C3 UART1 TX  → Display C3 GPIO 19 (RX)
 *    Main C3 UART1 RX  ← Display C3 GPIO 20 (TX, optional)
 */
#define UART_CMD_PORT   1       /* UART_NUM_1 */
#define UART_CMD_RX     20
#define UART_CMD_TX     21
#define UART_CMD_BAUD   115200

/* Start UART receiver task — calls face_set() on received commands */
void uart_cmd_init(void);
