// Philip Carr
// CS 171 Assignment 5
// November 27, 2018
// scene_read.h

#ifndef scene_read_h
#define scene_read_h

#include <stdio.h>
#include <string>
#include <math.h>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <stdexcept>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "halfedge.h"

using namespace std;
using namespace Eigen;

/**
 * Camera struct to store the camera's position, orientation, and perspective
 * in world space.
 */
struct Camera {
    /* Index 0 has the x-coordinate
     * Index 1 has the y-coordinate
     * Index 2 has the z-coordinate
     */
    float pos[3]; // position
    float ori_axis[3]; // orientation_axis

    /* Angle in degrees.
     */
    float ori_angle; // orientation_angle

    float n; // near
    float f; // far
    float l; // left
    float r; // right
    float t; // top
    float b; // bottom

    float fov; // Field of view of camera in degrees
    float aspect; // Aspect ratio of the camera
};

/* Point_Light struct for representing a point light source of a scene. The
 * position is represented in homogeneous coordinates rather than
 * the simple Cartesian coordinates normally used, since OpenGL requires a
 * w-coordinate be specified when the positions of the point lights are
 * specified. The Point_Light positions are specified in the set_lights
 * function.
 */
struct Point_Light {
    /* Index 0 has the x-coordinate
     * Index 1 has the y-coordinate
     * Index 2 has the z-coordinate
     * Index 3 has the w-coordinate
     */
    float pos[4]; // position

    /* Index 0 has the r-component
     * Index 1 has the g-component
     * Index 2 has the b-component
     */
    float color[3];

    /* This is our 'k' factor for attenuation as discussed in the lecture notes
     * and extra credit of Assignment 2.
     */
    float k; // attenuation_k
};

// /**
//  * Triple struct for storing vertices and vertex normal vectors (w-coordinates
//  * of vertices and vertex normal vectors do not need to be stored here since
//  * OpenGL handles them).
//  */
// struct Triple {
//     float x;
//     float y;
//     float z;
// };

/**
 * Transform struct to store a transformation. Each Transform struct contains
 * an array representation of a translation, rotation, or scaling
 * transformation.
 */
struct Obj_Transform {
    /* Type of transformation.
     * "t" = translation, "r" = rotation, "s" = scaling.
     */
    string type;

    /* For each array below,
     * Index 0 has the x-component
     * Index 1 has the y-component
     * Index 2 has the z-component
     */
    float components[3];

    // Angle in degrees.
    float rotation_angle;
};

/**
 * Object struct to store objects to render. Objects contain a name, vectors
 * of vertices and their corresponding vertex normal vectors, a vector of
 * associated transformations, and material properties used by the lighting
 * model.
 */
struct Object {
    string name;

    Mesh_Data *mesh_data;

    Mesh_Data *original_mesh_data;

    /* See the note above and the comments in the 'draw_objects' and
     * 'create_cubes' functions for details about these buffer vectors.
     */
    vector<Vertex> vertex_buffer;
    vector<Vec3f> normal_buffer;

    vector<Obj_Transform> transform_sets;

    vector<Matrix4f> t_transform_matrices;
    vector<Matrix4f> n_transform_matrices;

    vector<Vertex> transformed_vertices;
    vector<Vec3f> transformed_normals;

    /* Index 0 has the r-component
     * Index 1 has the g-component
     * Index 2 has the b-component
     */
    float ambient_reflect[3];
    float diffuse_reflect[3];
    float specular_reflect[3];

    float shininess;
};

/**
 * Return true if a given string is found in a vector of strings. Otherwise
 * return false.
 */
bool string_in_vector(const vector<string> &v, const string a);

/**
 * Return the index of a given string in a vector of strings if the given string
 * is found. Otherwise, return -1 if the given string is not found in the
 * vector.
 */
int string_index_in_vector(const vector<string> &v, const string a);

 /**
  * Return the vector normal to the surface defined by the given halfedge face.
  * (The returned vector is not normalized.)
  */
 Vec3f calc_face_normal(HEF *face);

 /**
  * Return the magnitude of a Vec3f struct.
  */
 float vec3f_magnitude(Vec3f *v);

 /**
  * From halfedge.h
  *
  * Return the area-weighted normal vector of a vertex by computing a sum of the
  * face normals adjacent to the given halfedge vertex weighted by their
  * respective face areas.
  */
 Vec3f calc_vertex_normal(HEV *vertex);

 /**
  * Return a Mesh_Data struct of an object's vertices and faces from a file.
  *
  * invalid_argument thrown if neither "v" nor "f" is found at beginning of a
  * non-empty line.
  */
 Mesh_Data* read_object_file(const string filename);

 /**
  * Set the given object's vertex and normal buffers (using the object's
  * mesh_data).
  */
 void update_object_buffers(Object &obj);

 /**
  * Assign each halfedge vertex in halfedge vertex vector to an index.
  */
 void index_vertices(vector<HEV*> *hevs);

 /**
  * Return the sum of all the face areas adjacent to a given halfedge's vertex.
  */
 float calc_neighbor_area_sum(HE *he);

 /**
  * Return the cotangent of the angle alpha given halfedge vertices vi and vj.
  * The angle alpha corresponds to the angle opposite to the flipped halfedge
  * face of the given halfedge start.
  */
 float calc_cot_alpha(HEV *vi, HEV *vj, HE *start);

 /**
  * Return the cotangent of the angle beta given halfedge vertices vi and vj.
  * The angle beta corresponds to the angle opposite to the halfedge face of the
  * given halfedge start.
  */
 float calc_cot_beta(HEV *vi, HEV *vj, HE *start);

 /**
  * Return the F operator for implicit fairing in matrix form.
  */
 SparseMatrix<float> build_F_operator(vector<HEV*> *hevs, const float h);

 /**
  * Solve the nonlinear matrix equations F x_h = x_0, F y_h = y_0, and
  * F z_h = z_0, and update the halfedge vertices with the vertices
  * (x_h, y_h, z_h) corresponding to the smoothed object.
  */
 void solve(vector<HEV*> *hevs, SparseMatrix<float> F);

 /**
  * Smooth the given object using implicit fairing with the given time step h.
  */
 void smooth_object(Object &obj, const float h);

 /**
  * Return an Object struct of an object corresponding to the object file with
  * the given filename, using halfedges to compute the vertex normals of the
  * object.
  */
 Object get_object_from_file(const string filename);

/**
 * Read the camera information from a file, starting at the "camera:" token.
 *
 * invalid_argument thrown if file cannot be opened.
 *
 * invalid_argument thrown if no "camera:" token is found.
 *
 * invalid_argument thrown if unknown camera information (not related to
 * position, orientation, or perspective) is found.
 */
Camera read_camera(const string filename);

/**
 * Return a vector of all the lights found in a scene description file.
 *
 * invalid_argument thrown if file cannot be opened.
 */
vector<Point_Light> read_lights(const string filename);

/**
 * Read all the objects (vertices, faces, and corresponding transformations)
 * from a file.
 *
 * invalid_argument thrown if file cannot be opened.
 *
 * invalid_argument thrown if no "objects:" token is found.
 *
 * invalid_argument thrown if neither "t", "r", "s", nor an object name is
 * found at beginning of a non-empty line.
 */
vector<Object> read_objects(const string filename);

#endif
