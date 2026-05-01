CROSS   := aarch64-linux-gnu-
CC      := $(CROSS)gcc

TARGET  := gru_behavior_test
SRCS    := gru_behavior_test.c
INCS    := -I../include
LIBS    := -L../lib -lcviruntime -lcvikernel -lm -Wl,-rpath,/mnt/system/lib
CFLAGS  := -O2 -Wall $(INCS)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $^ -o $@ $(LIBS)
	@echo "Built: $(TARGET)"

clean:
	rm -f $(TARGET)

.PHONY: clean
