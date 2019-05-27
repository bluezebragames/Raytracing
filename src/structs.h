// Philip Carr
// CS 171 Assignment 5
// November 27, 2018
// structs.h

#ifndef STRUCTS_H
#define STRUCTS_H

#include <vector>

struct Vec3f
{
	float x, y, z;
};

struct Vertex
{
    float x, y, z;
	// int index;
};

struct Face
{
    int idx1, idx2, idx3;
};

struct Mesh_Data
{
    std::vector<Vertex*> *vertices;
    std::vector<Face*> *faces;
};

#endif
