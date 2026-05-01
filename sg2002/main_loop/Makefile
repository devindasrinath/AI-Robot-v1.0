CROSS   := aarch64-linux-gnu-
CC      := $(CROSS)gcc

TARGET  := nexu_main
SRCS    := main.c inference.c uart_sg.c audio.c action.c
INCS    := -I../include -I.
LIBS    := -L../lib -lcviruntime -lcvikernel -lpthread -lm \
           -Wl,-rpath,/mnt/system/lib \
           -Wl,-rpath,/mnt/system/usr/lib \
           -Wl,-rpath,/mnt/system/usr/lib/3rd
CFLAGS  := -O2 -Wall $(INCS)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $^ -o $@ $(LIBS)
	@echo "Built: $(TARGET)"

clean:
	rm -f $(TARGET)

.PHONY: clean
