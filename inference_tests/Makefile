CC = gcc
CFLAGS = -Wall -I./model -I./cubeai_lib/include
LDFLAGS = -L./cubeai_lib/lib/static  -lruntime -lst_cmsis_nn -lcmsis-nn -lx86_cmsis -lm 

MODEL_DIR = model
BUILD = build

SRCS = main.c
MODEL_SRCS = $(wildcard $(MODEL_DIR)/*.c)

OBJS = $(SRCS:.c=.o) $(MODEL_SRCS:.c=.o)

TARGET = inference_test

all: $(TARGET)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(CC) $(OBJS) $(LDFLAGS) -o $(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)

run:
	./$(TARGET)
