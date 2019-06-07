# Philip Carr
# CS 171 Assignment 5
# November 27, 2018
# Makefile

CC = g++
FLAGS = -g -std=c++11 -o
FLAGS = -g -o

INCLUDE = -I/usr/include -I/usr/local/Cellar/libpng/1.6.36/include/libpng16
LIBDIR = -L/usr/local/lib -L/usr/local/Cellar/libpng/1.6.36/lib -lpng16
LIBS = -framework OpenGL -framework GLUT

EXENAME = raytrace

all: $(SOURCES)
	$(eval INCLUDE = -I/usr/X11R6/include -I/usr/include/GL -I/usr/include -isystem../)
	$(eval LIBDIR = -L/usr/X11R6/lib -L/usr/local/lib)
	$(eval SOURCES = *.cpp)
	$(eval LIBS = -lGLEW -lGL -lGLU -lglut -lm -lpng)

	$(CC) $(FLAGS) $(EXENAME) $(INCLUDE) $(LIBDIR) $(SOURCES) $(LIBS)

mac: $(SOURCES)
	$(CC) $(FLAGS) $(EXENAME) $(INCLUDE) $(LIBDIR) -DMAC $(SOURCES) $(LIBS)

clean:
	rm -f *.o $(EXENAME)

.PHONY: all clean
