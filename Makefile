CC = gcc
CFLAGS = -Wall -Wextra -O2
LDFLAGS = -lm -framework Accelerate

OBJS = main.o math_ops.o transformer.o

transformer: $(OBJS)
	$(CC) $(OBJS) -o transformer $(LDFLAGS)

main.o: main.c math_ops.h transformer.h
	$(CC) $(CFLAGS) -c main.c

math_ops.o: math_ops.c math_ops.h
	$(CC) $(CFLAGS) -c math_ops.c

transformer.o: transformer.c transformer.h math_ops.h
	$(CC) $(CFLAGS) -c transformer.c

clean:
	rm -f *.o transformer
