CROSS   := aarch64-linux-gnu-
CC      := $(CROSS)gcc

TARGET  := face_detect_test
SRCS    := face_detector.c face_detect_test.c
INCS    := -I../include
LIBS    := -L../lib -lcviruntime -lcvikernel -lm -Wl,-rpath,/mnt/system/lib
CFLAGS  := -O2 -Wall $(INCS)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $^ -o $@ $(LIBS)
	@echo "Built: $(TARGET)"

clean:
	rm -f $(TARGET)

.PHONY: clean
