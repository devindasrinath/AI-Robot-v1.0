#include "uart_sg.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <pthread.h>

UartEvents g_uart_events = {0};

static int s_fd = -1;

/* ── Open and configure UART ─────────────────────────────────────────────── */

static int uart_open(void)
{
    int fd = open(UART_ESP_DEV, O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd < 0) {
        perror("[uart_sg] open");
        return -1;
    }

    struct termios tty;
    tcgetattr(fd, &tty);
    cfsetispeed(&tty, B921600);
    cfsetospeed(&tty, B921600);
    tty.c_cflag  = CS8 | CREAD | CLOCAL;
    tty.c_iflag  = 0;
    tty.c_oflag  = 0;
    tty.c_lflag  = 0;
    tty.c_cc[VMIN]  = 0;
    tty.c_cc[VTIME] = 0;
    tcsetattr(fd, TCSANOW, &tty);
    tcflush(fd, TCIOFLUSH);

    printf("[uart_sg] Opened %s @ %d baud\n", UART_ESP_DEV, UART_ESP_BAUD);
    return fd;
}

/* ── RX thread — parses lines from ESP32-C3 ─────────────────────────────── */

static void parse_event(const char *line)
{
    if      (strcmp(line, "TOUCH:head")      == 0) g_uart_events.touch_head      = true;
    else if (strcmp(line, "TOUCH:body")      == 0) g_uart_events.touch_body      = true;
    else if (strcmp(line, "TOUCH:tail")      == 0) g_uart_events.touch_tail      = true;
    else if (strcmp(line, "COLLISION:front") == 0) g_uart_events.collision_front = true;
    else if (strcmp(line, "COLLISION:left")  == 0) g_uart_events.collision_left  = true;
    else if (strcmp(line, "COLLISION:right") == 0) g_uart_events.collision_right = true;
    else fprintf(stderr, "[uart_sg] unknown event: %s\n", line);
}

static void *rx_thread(void *arg)
{
    (void)arg;
    char buf[128];
    int  pos = 0;
    char ch;

    while (1) {
        int n = read(s_fd, &ch, 1);
        if (n <= 0) {
            usleep(5000);  /* 5ms poll when no data */
            continue;
        }
        if (ch == '\n') {
            buf[pos] = '\0';
            /* Strip trailing \r */
            if (pos > 0 && buf[pos - 1] == '\r') buf[--pos] = '\0';
            if (pos > 0) parse_event(buf);
            pos = 0;
        } else if (ch != '\r') {
            if (pos < (int)sizeof(buf) - 1)
                buf[pos++] = ch;
        }
    }
    return NULL;
}

/* ── Public API ──────────────────────────────────────────────────────────── */

int uart_sg_init(void)
{
    s_fd = uart_open();
    if (s_fd < 0) {
        fprintf(stderr, "[uart_sg] WARNING: ESP32-C3 UART unavailable — "
                        "running without hardware\n");
        return 0;  /* non-fatal — continue without UART */
    }

    pthread_t tid;
    pthread_create(&tid, NULL, rx_thread, NULL);
    pthread_detach(tid);
    return 0;
}

void uart_sg_send(const char *cmd)
{
    if (s_fd < 0) return;  /* no hardware connected */
    write(s_fd, cmd, strlen(cmd));
}
