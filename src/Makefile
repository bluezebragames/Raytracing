# Philip Carr
# CS 171 Assignment 5
# November 27, 2018
# Makefile

CC = g++
FLAGS = -g -std=c++11 -o

INCLUDE = -I/usr/X11R6/include -I/usr/include/GL -I/usr/include -isystem../
LIBDIR = -L/usr/X11R6/lib -L/usr/local/lib
SOURCES = *.cpp *.h *.hpp
LIBS = -lGLEW -lGL -lGLU -lglut -lm -lpng

EXENAME = smooth

all: $(SOURCES)
	$(CC) $(FLAGS) $(EXENAME) $(INCLUDE) $(LIBDIR) $(SOURCES) $(LIBS)

clean:
	rm -f *.o $(EXENAME)

.PHONY: all clean
