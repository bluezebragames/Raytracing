/* CS/CNS 171
 * Written by Kevin (Kevli) Li (Class of 2016)
 * Originally for Fall 2014
 *
 * This OpenGL demo code is supposed to introduce you to the OpenGL syntax and
 * to good coding practices when writing programs in OpenGL.
 *
 * The example syntax and code organization in this file should hopefully be
 * good references for you to write your own OpenGL code.
 *
 * The advantage of OpenGL is that it turns a lot of complicated procedures
 * (such as the lighting and shading computations in Assignment 2) into simple
 * calls to built-in library functions. OpenGL also provides an easy way to
 * make mouse and keyboard user interfaces, allowing you to make programs that
 * actually let you interact with the graphics instead of just generating
 * static images. OpenGL is in general a nice tool for when you want to make a
 * quick-and-dirty graphics program.
 *
 * Keep in mind that this demo code uses OpenGL 3.0. 3.0 is not the newest
 * version of OpenGL, but it is stable; and it contains all the necessary
 * functionality for this class. Most of the syntax in 3.0 carries over to
 * the newer versions, so you should still be able to use more modern OpenGL
 * without too much difficulty after this class. The main difference between
 * 3.0 and the newer versions is that 3.0 depends on glut, which has been
 * deprecated on Mac OS.
 *
 * This demo does not cover the OpenGL Shading Language (GLSL for short).
 * GLSL will be covered in a future demo and assignment.
 *
 * Note that if you are looking at this code before having completed
 * Assignments 1 and 2, then you will probably have a hard time understanding
 * a lot of what is going on.
 *
 * The overall idea of what this program does is given on the
 * "System Recommendations and Installation Instructions" page of the class
 * website.
 */


/* The following 2 headers contain all the main functions, data structures, and
 * variables that allow for OpenGL development.
 */
//#include <OpenGL/glew.h>
#include <GLUT/glut.h>

/* You will almost always want to include the math library. For those that do
 * not know, the '_USE_MATH_DEFINES' line allows you to use the syntax 'M_PI'
 * to represent pi to double precision in C++. OpenGL works in degrees for
 * angles, so converting between degrees and radians is a common task when
 * working in OpenGL.
 *
 * Besides the use of 'M_PI', the trigometric functions also show up a lot in
 * graphics computations.
 */
#include <math.h>
#define _USE_MATH_DEFINES

/* iostream and vector are standard libraries that are just generally useful.
 */
#include <iostream>
#include <vector>

/* For sanity checks.
 */
#include <assert.h>

/* For keeping track of the directory where the models are stored.
 */
#include <libgen.h>

/* For file I/O.
 */
#include <fstream>

/* Used to keep track of the object names.
 */
#include <map>

// Use halfedges for vertex normals since they're not given in the object files
#include "halfedge.h"

// For matrix solving -- the Laplacian is very sparse
#include "Eigen/Dense"
#include "Eigen/Sparse"


using namespace std;

///////////////////////////////////////////////////////////////////////////////////////////////////

/* The following are function prototypes for the functions that you will most
 * often write when working in OpenGL.
 *
 * Details on the functions will be given in their respective implementations
 * further below.
 */

void init(char*);
void reshape(int width, int height);
void display(void);

void input(char*);
void input_camera(ifstream& fin);

void init_lights();
void set_lights();
void draw_objects();

void mouse_pressed(int button, int state, int x, int y);
void mouse_moved(int x, int y);

float deg2rad(float);
float rad2deg(float);

///////////////////////////////////////////////////////////////////////////////////////////////////

/* The following structs do not involve OpenGL, but they are useful ways to
 * store information needed for rendering,
 *
 * After Assignment 2, the 3D shaded surface renderer assignment, you should
 * have a fairly intuitive understanding of what these structs represent.
 */


/* The following struct is used for representing a point light.
 *
 * Note that the position is represented in homogeneous coordinates rather than
 * the simple Cartesian coordinates that we would normally use. This is because
 * OpenGL requires us to specify a w-coordinate when we specify the positions
 * of our point lights. We specify the positions in the 'input' function.
 */
struct Point_Light
{
    /* Index 0 has the x-coordinate
     * Index 1 has the y-coordinate
     * Index 2 has the z-coordinate
     * Index 3 has the w-coordinate
     */
    float position[4];

    /* Index 0 has the r-component
     * Index 1 has the g-component
     * Index 2 has the b-component
     */
    float color[3];

    /* This is our 'k' factor for attenuation as discussed in the lecture notes
     * and extra credit of Assignment 2.
     */
    float attenuation_k;
};

/* The following struct is used for representing points and normals in world
 * coordinates.
 *
 * Notice how we are using this struct to represent points, but the struct
 * lacks a w-coordinate. Fortunately, OpenGL will handle all the complications
 * with the homogeneous component for us when we have it process the points.
 * We do not actually need to keep track of the w-coordinates of our points
 * when working in OpenGL.
 */
typedef Vertex Triple;
/*

{
    float x;
    float y;
    float z;
};
*/

/* The following struct is used for storing a set of transformations.
 * Please note that this structure assumes that our scenes will give
 * sets of transformations in the form of transltion -> rotation -> scaling.
 * Obviously this will not be the case for your scenes. Keep this in
 * mind when writing your own programs.
 *
 * Note that we do not need to use matrices this time to represent the
 * transformations. This is because OpenGL will handle all the matrix
 * operations for us when we have it apply the transformations. All we
 * need to do is supply the parameters.
 */
struct Transform
{
    /* Type 0: translation
     * Type 1: rotation
     * Type 2: scaling
     */
    int transform_type;

    /* Index 0 has the x-component
     * Index 1 has the y-component
     * Index 2 has the z-component
     */
    float components[3];

    /* Angle in degrees.
     */
    float rotation_angle;
};

/* The following struct is used to represent objects.
 *
 * The main things to note here are the 'vertex_buffer' and 'normal_buffer'
 * vectors.
 *
 * You will see later in the 'draw_objects' function that OpenGL requires
 * us to supply it all the faces that make up an object in one giant
 * "vertex array" before it can render the object. The faces are each specified
 * by the set of vertices that make up the face, and the giant "vertex array"
 * stores all these sets of vertices consecutively. Our "vertex_buffer" vector
 * below will be our "vertex array" for the object.
 *
 * As an example, let's say that we have a cube object. A cube has 6 faces,
 * each with 4 vertices. Each face is going to be represented by the 4 vertices
 * that make it up. We are going to put each of these 4-vertex-sets one by one
 * into 1 large array. This gives us an array of 36 vertices. e.g.:
 *
 * [face1vertex1, face1vertex2, face1vertex3, face1vertex4,
 *  face2vertex1, face2vertex2, face2vertex3, face2vertex4,
 *  face3vertex1, face3vertex2, face3vertex3, face3vertex4,
 *  face4vertex1, face4vertex2, face4vertex3, face4vertex4,
 *  face5vertex1, face5vertex2, face5vertex3, face5vertex4,
 *  face6vertex1, face6vertex2, face6vertex3, face6vertex4]
 *
 * This array of 36 vertices becomes our 'vertex_array'.
 *
 * While it may be obvious to us that some of the vertices in the array are
 * repeats, OpenGL has no way of knowing this. The redundancy is necessary
 * since OpenGL needs the vertices of every face to be explicitly given.
 *
 * The 'normal_buffer' stores all the normals corresponding to the vertices
 * in the 'vertex_buffer'. With the cube example, since the "vertex array"
 * has "36" vertices, the "normal array" also has "36" normals.
 */
struct Object
{
    // Keep track of the HEVs and HEFs for constructing the Laplacian
    vector<HEV*> *hevs;
    vector<HEF*> *hefs;

    // Keep track of the current mesh for recomputing everything after smoothing
    Mesh_Data *mesh;

    /* See the note above and the comments in the 'draw_objects' and
     * 'create_cubes' functions for details about these buffer vectors.
     */
    vector<Triple> vertex_buffer;
    vector<Triple> normal_buffer;

    vector<Transform> transforms;

    /* Index 0 has the r-component
     * Index 1 has the g-component
     * Index 2 has the b-component
     */
    float ambient_reflect[3];
    float diffuse_reflect[3];
    float specular_reflect[3];

    float shininess;
};

/* Quaternion implementation: q = s + xi + yj + zk
 * which we represent as s and a triple v = (x, y, z).
 */
typedef struct Quaternion {
    float s;
    Triple v;
} Quaternion;

Quaternion identity() {
    Quaternion i;
    i.s = 1;
    i.v.x = 0;
    i.v.y = 0;
    i.v.z = 0;
    return i;
}


///////////////////////////////////////////////////////////////////////////////////////////////////

/* The following are the typical camera specifications and parameters. In
 * general, it is a better idea to keep all this information in a camera
 * struct, like how we have been doing it in Assignemtns 1 and 2. However,
 * if you only have one camera for the scene, and all your code is in one
 * file (like this one), then it is sometimes more convenient to just have
 * all the camera specifications and parameters as global variables.
 */

/* Index 0 has the x-coordinate
 * Index 1 has the y-coordinate
 * Index 2 has the z-coordinate
 */
float cam_position[3];
float cam_orientation_axis[3];
float cam_orientation_angle;
float near_param, far_param, left_param, right_param, top_param, bottom_param;

///////////////////////////////////////////////////////////////////////////////////////////////////

/* Self-explanatory lists of lights and objects.
 */

vector<Point_Light> lights;
vector<Object> objects;

map<string, Mesh_Data> mesh_name_to_mesh; // Given a mesh name, return the actual mesh in O(log(# of meshs))

///////////////////////////////////////////////////////////////////////////////////////////////////

/* The following are parameters for implementing an arcball for rotating the models in a scene. The variables will make more sense when explained in
 * context, so you should just look at the 'mousePressed' and 'mouseMoved'
 * functions for the details.
 */

int xres, yres;
int mouse_x_start, mouse_y_start;
int mouse_x_current, mouse_y_current;
bool is_pressed = false;

Quaternion last_rotation = identity(), curr_rotation = identity();
double rot[16];

// Time step for implicit fairing
double h;

///////////////////////////////////////////////////////////////////////////////////////////////////

/* From here on are all the function implementations.
 */


/*
 * Quaternion functionality
 */
Triple add(Triple a, Triple b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

Triple scale(float c, Triple a) {
    a.x *= c;
    a.y *= c;
    a.z *= c;
    return a;
}

Triple subtract(Triple a, Triple b) {
    return add(a, scale(-1, b));
}

Triple cross(Triple a, Triple b) {
    Triple c;
    c.x = a.y*b.z - a.z*b.y;
    c.y = a.z*b.x - a.x*b.z;
    c.z = a.x*b.y - a.y*b.x;
    return c;
}

double dot(Triple a, Triple b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

float norm(Triple a) {
    return sqrt(dot(a,a));
}

Quaternion add(Quaternion a, Quaternion b) {
    a.s += b.s;
    a.v = add(a.v, b.v);
    return a;
}

Quaternion scale(float c, Quaternion a) {
    a.s *= c;
    a.v = scale(c, a.v);
    return a;
}

Quaternion subtract(Quaternion a, Quaternion b) {
    return add(a, scale(-1, b));
}

Quaternion product(Quaternion a, Quaternion b) {
    Quaternion c;
    c.s = a.s*b.s - dot(a.v, b.v);
    c.v = add(add(scale(a.s, b.v), scale(b.s, a.v)), cross(a.v, b.v));
    return c;
}

Quaternion conjugate(Quaternion a) {
    a.v = scale(-1, a.v);
    return a;
}

float norm(Quaternion a) {
    return sqrt(a.s*a.s + dot(a.v, a.v));
}

Quaternion compute_rotation_quaternion(float px, float py, float ppx, float ppy) {
    Triple pndc, ppndc;
    pndc.x = px / (xres/2) - 1;
    pndc.y = -1 * (py / (yres/2) - 1);
    pndc.z = 0;
    if(pndc.x * pndc.x + pndc.y * pndc.y <= 1) {
        pndc.z = sqrt(1 - (pndc.x * pndc.x + pndc.y * pndc.y));
    }
    ppndc.x = ppx / (xres/2) - 1;
    ppndc.y = -1 * (ppy / (yres/2) - 1);
    ppndc.z = 0;
    if(ppndc.x * ppndc.x + ppndc.y * ppndc.y <= 1) {
        ppndc.z = sqrt(1 - (ppndc.x * ppndc.x + ppndc.y * ppndc.y));
    }

    float theta = acos(min(1., dot(pndc, ppndc) / (norm(pndc) * norm(ppndc))));

    Triple u = cross(pndc, ppndc);
    u = scale(1 / norm(u), u);

    Quaternion r;
    r.s = cos(theta / 2);
    r.v = scale(sin(theta/2), u);
    return r;
}

double* get_current_rotation() {
    Quaternion q = product(last_rotation, curr_rotation);
    float qs = q.s;
    float qx = q.v.x;
    float qy = q.v.y;
    float qz = q.v.z;
    float qx2 = qx * qx;
    float qy2 = qy * qy;
    float qz2 = qz * qz;

    // Top row
    rot[0] = 1 - 2*qy2 - 2*qz2;
    rot[1] = 2 * (qx*qy - qz*qs);
    rot[2] = 2 * (qx*qz + qy*qs);
    rot[3] = 0;

    // Second row
    rot[4] = 2 * (qx*qy + qz*qs);
    rot[5] = 1 - 2*qx2 - 2*qz2;
    rot[6] = 2 * (qy*qz - qx*qs);
    rot[7] = 0;

    // Third row
    rot[8] = 2 * (qx*qz - qy*qs);
    rot[9] = 2 * (qy*qz + qx*qs);
    rot[10] = 1 - 2*qx2 - 2*qy2;
    rot[11] = 0;

    // Bottom row
    rot[12] = 0;
    rot[13] = 0;
    rot[14] = 0;
    rot[15] = 1;

    return rot;
}


// Given a halfedge vertex, compute the normal at that vertex by using HEs
Triple calc_vertex_normal(HEV *vertex) {
    Triple normal;
    normal.x = 0;
    normal.y = 0;
    normal.z = 0;

    HE* he = vertex->out; // get outgoing halfedge from given vertex

    do {
        HEF *f = he->face;
        HEV hev1 = *f->edge->vertex;
        HEV hev2 = *f->edge->next->vertex;
        HEV hev3 = *f->edge->next->next->vertex;
        Vertex v1; v1.x = hev1.x; v1.y = hev1.y; v1.z = hev1.z;
        Vertex v2; v2.x = hev2.x; v2.y = hev2.y; v2.z = hev2.z;
        Vertex v3; v3.x = hev3.x; v3.y = hev3.y; v3.z = hev3.z;
        // compute the normal of the plane of the face: cross(v2-v1, v3-v1)
        Triple face_normal = cross(add(v2, scale(-1, v1)), add(v3, scale(-1, v1)));
        // compute the area of the triangular face
        double face_area = 0.5 * sqrt(dot(face_normal, face_normal));
        face_normal = scale(1.0 / sqrt(dot(face_normal, face_normal)), face_normal);

        // accummulate onto our normal vector
        normal.x += face_normal.x * face_area;
        normal.y += face_normal.y * face_area;
        normal.z += face_normal.z * face_area;

        // gives us the halfedge to the next adjacent vertex
        he = he->flip->next;
    } while(he != vertex->out);

    vertex->normal = scale(1.0 / sqrt(dot(normal, normal)), normal);
}


//
void index_vertices(vector<HEV*> *vertices) {
    for( int i = 1; i < vertices->size(); ++i ) // start at 1 because obj files are 1-indexed
        vertices->at(i)->index = i; // assign each vertex an index
}


// function to calculate the cotangent of an angle, given as three vertices
// with the one we're calculating in the middle.
double calculate_cot(HEV *v1, HEV *v2, HEV *v3) {
    Triple t1, t2, t3;
    t1.x = v1->x;
    t1.y = v1->y;
    t1.z = v1->z;
    t2.x = v2->x;
    t2.y = v2->y;
    t2.z = v2->z;
    t3.x = v3->x;
    t3.y = v3->y;
    t3.z = v3->z;

    Triple side1, side2;
    side1 = add(t1, scale(-1, t2));
    side2 = add(t3, scale(-1, t2));

    double numerator = dot(side1, side2);
    double denominator = sqrt(dot(cross(side1, side2), cross(side1, side2)));
    return numerator / denominator;
}

double calculate_tri_area(HEV *v1, HEV *v2, HEV *v3) {
    Triple t1, t2, t3;
    t1.x = v1->x;
    t1.y = v1->y;
    t1.z = v1->z;
    t2.x = v2->x;
    t2.y = v2->y;
    t2.z = v2->z;
    t3.x = v3->x;
    t3.y = v3->y;
    t3.z = v3->z;

    Triple side1, side2;
    side1 = add(t1, scale(-1, t2));
    side2 = add(t3, scale(-1, t2));

    Triple normal = cross(side1, side2);
    return 0.5 * sqrt(dot(normal, normal));
}



// function to construct our F operator in matrix form
// F = I - h(delta)
Eigen::SparseMatrix<double> build_F_operator(std::vector<HEV*> *vertices) {
    index_vertices(vertices); // assign each vertex an index

    // recall that due to 1-indexing of obj files, index 0 of our list doesn’t actually contain a vertex
    int num_vertices = vertices->size() - 1;

    // initialize a sparse matrix to represent our F operator
    Eigen::SparseMatrix<double> F(num_vertices, num_vertices);

    // reserve room for 7 non-zeros per row of B
    F.reserve(Eigen::VectorXi::Constant(num_vertices, 7));

    for(int i = 1; i < vertices->size(); ++i) {
        HEV v_i = *vertices->at(i);
        HE *he = vertices->at(i)->out;

        double A = 0.0; // Area of all of the faces around v_i
        // First loop to calculate total adjacent triangle area
        do { // iterate over all vertices adjacent to v_i
            int j = he->next->vertex->index; // get index of adjacent vertex to v_i

            // Triangle 1: vertices of v_i, v_j, and he->next->vertex
            HEV *v1 = vertices->at(i);
            HEV *v2 = vertices->at(j);
            HEV *v3 = he->next->next->vertex;
            HEV *v4 = he->flip->next->next->vertex;

            double curr_A = 0.5 * (calculate_tri_area(v1, v3, v2) + calculate_tri_area(v2, v4, v1));
            A += curr_A;

            he = he->flip->next;
        } while(he != vertices->at(i)->out);

        if (A >= 0.000001) {
            he = vertices->at(i)->out;

            double cot_sum = 0.0;
            // Second loop to populate matrix row
            do { // iterate over all vertices adjacent to v_i
                int j = he->next->vertex->index; // get index of adjacent vertex to v_i

                // Triangle 1: vertices of v_i, v_j, and he->next->vertex
                HEV *v1 = vertices->at(i);
                HEV *v2 = vertices->at(j);
                HEV *v3 = he->next->next->vertex;
                HEV *v4 = he->flip->next->next->vertex;
                double cot_alpha = calculate_cot(v1, v3, v2);
                double cot_beta = calculate_cot(v2, v4, v1);

                // fill the j-th slot of row i of our B matrix with appropriate value
                F.insert(i-1, j-1) = 1 / (2*A) * (cot_alpha + cot_beta);
                cot_sum += cot_alpha + cot_beta;

                he = he->flip->next;
            } while(he != vertices->at(i)->out);

            // Diagonal matrix entry
            F.insert(i-1, i-1) = -1 / (2*A) * cot_sum;
        }
    }
    F.makeCompressed(); // optional; tells Eigen to more efficiently store our sparse matrix

    Eigen::SparseMatrix<double> I(num_vertices, num_vertices);
    I.setIdentity();
    return (I - h * F);
}


void smooth(Object* curr_object) {
    vector<HEV*> *vertices = curr_object->hevs;
    // get our matrix representation of F
    Eigen::SparseMatrix<double> F = build_F_operator(vertices);

    // initialize Eigen’s sparse solver
    Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> > solver;

    // the following two lines essentially tailor our solver to our operator B
    solver.analyzePattern(F);
    solver.factorize(F);

    int num_vertices = curr_object->mesh->vertices->size() - 1;

    // initialize our vector representation of x_0, y_0, and z_0
    Eigen::VectorXd x_0_vector(num_vertices);
    Eigen::VectorXd y_0_vector(num_vertices);
    Eigen::VectorXd z_0_vector(num_vertices);
    for(int i = 1; i < curr_object->mesh->vertices->size(); ++i) {
        x_0_vector(i - 1) = curr_object->mesh->vertices->at(i)->x;
        y_0_vector(i - 1) = curr_object->mesh->vertices->at(i)->y;
        z_0_vector(i - 1) = curr_object->mesh->vertices->at(i)->z;
    }

    // have Eigen solve for our _h vectors
    Eigen::VectorXd x_h_vector(num_vertices);
    Eigen::VectorXd y_h_vector(num_vertices);
    Eigen::VectorXd z_h_vector(num_vertices);
    x_h_vector = solver.solve(x_0_vector);
    y_h_vector = solver.solve(y_0_vector);
    z_h_vector = solver.solve(z_0_vector);

    for(int i = 1; i < vertices->size(); ++i) {
        curr_object->mesh->vertices->at(i)->x = x_h_vector(i - 1);
        curr_object->mesh->vertices->at(i)->y = y_h_vector(i - 1);
        curr_object->mesh->vertices->at(i)->z = z_h_vector(i - 1);
    }

    // Get the new normals
    curr_object->hevs->clear();
    curr_object->hefs->clear();
    build_HE(curr_object->mesh, curr_object->hevs, curr_object->hefs);
    for(int i = 1; i<curr_object->hevs->size(); ++i) {
        calc_vertex_normal(curr_object->hevs->at(i));
    }

    curr_object->vertex_buffer.clear();
    curr_object->normal_buffer.clear();
    for(int i = 0; i<curr_object->mesh->faces->size(); ++i) {
        Face *curr_face = curr_object->mesh->faces->at(i);
        curr_object->vertex_buffer.push_back(*curr_object->mesh->vertices->at(curr_face->idx1));
        curr_object->vertex_buffer.push_back(*curr_object->mesh->vertices->at(curr_face->idx2));
        curr_object->vertex_buffer.push_back(*curr_object->mesh->vertices->at(curr_face->idx3));
        // Normals generated from HE mesh
        curr_object->normal_buffer.push_back(curr_object->hevs->at(curr_face->idx1)->normal);
        curr_object->normal_buffer.push_back(curr_object->hevs->at(curr_face->idx2)->normal);
        curr_object->normal_buffer.push_back(curr_object->hevs->at(curr_face->idx3)->normal);
    }
}


/* 'init' function:
 *
 * As you would expect, the 'init' function initializes and sets up the
 * program. It should always be called before anything else.
 *
 * Writing an 'init' function is not required by OpenGL. If you wanted to, you
 * could just put all your initializations in the beginning of the 'main'
 * function instead. However, doing so is bad style; it is cleaner to have all
 * your initializations contained within one function.
 *
 * Before we go into the function itself, it is important to mention that
 * OpenGL works like a state machine. It will do different procedures depending
 * on what state it is in.
 *
 * For instance, OpenGL has different states for its shading procedure. By
 * default, OpenGL is in "flat shading state", meaning it will always use flat
 * shading when we tell it to render anything. With some syntax, we can change
 * the shading procedure from the "flat shading state" to the "Gouraud shading
 * state", and then OpenGL will render everything using Gouraud shading.
 *
 * The most important task of the 'init' function is to set OpenGL to the
 * states that we want it to be in.
 */
void init(char* scene_description_file_name)
{
    input(scene_description_file_name);

    /* The following line of code tells OpenGL to use "smooth shading" (aka
     * Gouraud shading) when rendering.
     *
     * Yes. This is actually all you need to do to use Gouraud shading in
     * OpenGL (besides providing OpenGL the vertices and normals to render).
     * Short and sweet, right?
     *
     * If you wanted to tell OpenGL to use flat shading at any point, then you
     * would use the following line:

       glShadeModel(GL_FLAT);

     * Phong shading unfortunately requires GLSL, so it will be covered in a
     * later demo.
     */
    glShadeModel(GL_SMOOTH);

    /* The next line of code tells OpenGL to use "culling" when rendering. The
     * line right after it tells OpenGL that the particular "culling" technique
     * we want it to use is backface culling.
     *
     * "Culling" is actually a generic term for various algorithms that
     * prevent the rendering process from trying to render unnecessary
     * polygons. Backface culling is the most commonly used method, but
     * there also exist other, niche methods like frontface culling.
     */
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    /* The following line tells OpenGL to use depth buffering when rendering.
     */
    glEnable(GL_DEPTH_TEST);

     /* The following line tells OpenGL to automatically normalize our normal
     * vectors before it passes them into the normal arrays discussed below.
     * This is required for correct lighting, but it also slows down our
     * program. An alternative to this is to manually scale the normal vectors
     * to correct for each scale operation we call. For instance, if we were
     * to scale an object by 3 (via glScalef() discussed below), then
     * OpenGL would scale the normals of the object by 1/3, as we would
     * expect from the inverse normal transform. But since we need unit
     * normals for lighting, we would either need to enable GL_NORMALIZE
     * or manually scale our normals by 3 before passing them into the
     * normal arrays; this is of course to counteract the 1/3 inverse
     * scaling when OpenGL applies the normal transforms. Enabling GL_NORMALIZE
     * is more convenient, but we sometimes don't use it if it slows down
     * our program too much.
     */
    glEnable(GL_NORMALIZE);

    /* The following two lines tell OpenGL to enable its "vertex array" and
     * "normal array" functionality. More details on these arrays are given
     * in the comments on the 'Object' struct and the 'draw_objects' and
     * 'create_objects' functions.
     */
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);

    /* The next 4 lines work with OpenGL's two main matrices: the "Projection
     * Matrix" and the "Modelview Matrix". Only one of these two main matrices
     * can be modified at any given time. We specify the main matrix that we
     * want to modify with the 'glMatrixMode' function.
     *
     * The Projection Matrix is the matrix that OpenGL applies to points in
     * camera space. For our purposes, we want the Projection Matrix to be
     * the perspective projection matrix, since we want to convert points into
     * NDC after they are in camera space.
     *
     * The line of code below:
     */
    glMatrixMode(GL_PROJECTION);
    /* ^tells OpenGL that we are going to modify the Projection Matrix. From
     * this point on, any matrix comamnds we give OpenGL will affect the
     * Projection Matrix. For instance, the line of code below:
     */
    glLoadIdentity();
    /* ^tells OpenGL to set the current main matrix (which is the Projection
     * Matrix right now) to the identity matrix. Then, the next line of code:
     */
    glFrustum(left_param, right_param,
              bottom_param, top_param,
              near_param, far_param);
    /* ^ tells OpenGL to create a perspective projection matrix using the
     * given frustum parameters. OpenGL then post-multiplies the current main
     * matrix (the Projection Matrix) with the created matrix. i.e. let 'P'
     * be our Projection Matrix and 'F' be the matrix created by 'glFrustum'.
     * Then, after 'F' is created, OpenGL performs the following operation:
     *
     * P = P * F
     *
     * Since we had set the Projection Matrix to the identity matrix before the
     * call to 'glFrustum', the above multiplication results in the Projection
     * Matrix being the perspective projection matrix, which is what we want.
     */

    /* The Modelview Matrix is the matrix that OpenGL applies to untransformed
     * points in world space. OpenGL applies the Modelview Matrix to points
     * BEFORE it applies the Projection Matrix.
     *
     * Thus, for our purposes, we want the Modelview Matrix to be the overall
     * transformation matrix that we apply to points in world space before
     * applying the perspective projection matrix. This means we would need to
     * factor in all the individual object transformations and the camera
     * transformations into the Modelview Matrix.
     *
     * The following line of code tells OpenGL that we are going to modify the
     * Modelview Matrix. From this point on, any matrix commands we give OpenGL
     * will affect the Modelview Matrix.
     *
     * We generally modify the Modelview Matrix in the 'display' function,
     * right before we tell OpenGL to render anything. See the 'display'
     * for details.
     */
    glMatrixMode(GL_MODELVIEW);

    /* The next line calls our function that tells OpenGL to initialize some
     * lights to represent our Point Light structs. Further details will be
     * given in the function itself.
     *
     * The reason we have this procedure as a separate function is to make
     * the code more organized.
     */
    init_lights();
}

/* 'reshape' function:
 *
 * You will see down below in the 'main' function that whenever we create a
 * window in OpenGL, we have to specify a function for OpenGL to call whenever
 * the window resizes. We typically call this function 'reshape' or 'resize'.
 *
 * The 'reshape' function is supposed to tell your program how to react
 * whenever the program window is resized. It is also called in the beginning
 * when the window is first created. You can think of the first call to
 * 'reshape' as an initialization phase and all subsequent calls as update
 * phases.
 *
 * Anything that needs to know the dimensions of the program window should
 * be initialized and updated in the 'reshape' function. You will see below
 * that we use the 'reshape' function to initialize and update the conversion
 * scheme between NDC and screen coordinates as well as the mouse interaction
 * parameters.
 */
void reshape(int width, int height)
{
    /* The following two lines of code prevent the width and height of the
     * window from ever becoming 0 to prevent divide by 0 errors later.
     * Typically, we let 1x1 square pixel be the smallest size for the window.
     */
    height = (height == 0) ? 1 : height;
    width = (width == 0) ? 1 : width;

    /* The 'glViewport' function tells OpenGL to determine how to convert from
     * NDC to screen coordinates given the dimensions of the window. The
     * parameters for 'glViewport' are (in the following order):
     *
     * - int x: x-coordinate of the lower-left corner of the window in pixels
     * - int y: y-coordinate of the lower-left corner of the window in pixels
     * - int width: width of the window
     * - int height: height of the window
     *
     * We typically just let the lower-left corner be (0,0).
     *
     * After 'glViewport' is called, OpenGL will automatically know how to
     * convert all our points from NDC to screen coordinates when it tries
     * to render them.
     */
    glViewport(0, 0, width, height);

    /* The following line tells OpenGL that our program window needs to
     * be re-displayed, meaning everything that was being displayed on
     * the window before it got resized needs to be re-rendered.
     */
    glutPostRedisplay();
}

/* 'display' function:
 *
 * You will see down below in the 'main' function that whenever we create a
 * window in OpenGL, we have to specify a function for OpenGL to call whenever
 * it wants to render anything. We typically name this function 'display' or
 * 'render'.
 *
 * The 'display' function is supposed to handle all the processing of points
 * in world and camera space.
 */
void display(void)
{
    /* The following line of code is typically the first line of code in any
     * 'display' function. It tells OpenGL to reset the "color buffer" (which
     * is our pixel grid of RGB values) and the depth buffer.
     *
     * Resetting the "color buffer" is equivalent to clearing the program
     * window so that it only displays a black background. This allows OpenGL
     * to render a new scene onto the window without having to deal with the
     * remnants of the previous scene.
     *
     * Resetting the depth buffer sets all the values in the depth buffer back
     * to a very high number. This allows the depth buffer to be reused for
     * rendering a new scene.
     */
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    /* With the program window cleared, OpenGL is ready to render a new scene.
     * Of course, before we can render anything correctly, we need to make all
     * the appropriate camera and object transformations to our coordinate
     * space.
     *
     * Recall that the 'init' function used the glMatrixMode function to put
     * OpenGL into a state where we can modify its Modelview Matrix. Also
     * recall that we want the Modelview Matrix to be the overall transform-
     * ation matrix that we apply to points in world space before applying the
     * perspective projection matrix. This means that we need to factor in all
     * the individual object transformations and the camera transformations
     * into the Modelview Matrix.
     *
     * To do so, our first step is to "reset" the Modelview Matrix by setting it
     * to the identity matrix:
     */
    glLoadIdentity();
    /* Now, if you recall, for a given object, we want to FIRST multiply the
     * coordinates of its points by the translations, rotations, and scalings
     * applied to the object and THEN multiply by the inverse camera rotations
     * and translations.
     *
     * HOWEVER, OpenGL modifies the Modelview Matrix using POST-MULTIPLICATION.
     * This means that if were to specify to OpenGL a matrix modification 'A',
     * then letting the Modelview Matrix be 'M', OpenGL would perform the
     * following operation:
     *
     * M = M * A
     *
     * So, for instance, if the Modelview Matrix were initialized to the
     * identity matrix 'I' and we were to specify a translation 'T' followed by
     * a rotation 'R' followed by a scaling 'S' followed by the inverse camera
     * transform 'C', then the Modelview Matrix is modified in the following
     * order:
     *
     * M = I * T * R * S * C
     *
     * Then, when OpenGL applies the Modelview Matrix to a point 'p', we would
     * get the following multiplication:
     *
     * M * p = I * T * R * S * C * p
     *
     * ^ So the camera transformation ends up being applied first even though
     * it was specified last. This is not what we want. What we want is
     * something like this:
     *
     * M * p = C * T * R * S * I * p
     *
     * Hence, to correctly transform a point, we actually need to FIRST specify
     * the inverse camera rotations and translations and THEN specify the
     * translations, rotations, and scalings applied to an object.
     *
     * We start by specifying any camera rotations caused by the mouse. We do
     * so by using the 'glRotatef' function, which takes the following parameters
     * in the following order:
     *
     * - float angle: rotation angle in DEGREES
     * - float x: x-component of rotation axis
     * - float y: y-component of rotation axis
     * - float z: z-component of rotation axis
     *
     * The 'glRotatef' function tells OpenGL to create a rotation matrix using
     * the given angle and rotation axis.
     */
    /* Our next step is to specify the inverse rotation of the camera by its
     * orientation angle about its orientation axis:
     */
    glRotatef(-cam_orientation_angle,
              cam_orientation_axis[0], cam_orientation_axis[1], cam_orientation_axis[2]);
    /* We then specify the inverse translation of the camera by its position using
     * the 'glTranslatef' function, which takes the following parameters in the
     * following order:
     *
     * - float x: x-component of translation vector
     * - float y: x-component of translation vector
     * - float z: x-component of translation vector
     */
    glTranslatef(-cam_position[0], -cam_position[1], -cam_position[2]);
    /* ^ And that should be it for the camera transformations.
     */

    /* Our next step is to set up all the lights in their specified positions.
     * Our helper function, 'set_lights' does this for us. See the function
     * for more details.
     *
     * The reason we have this procedure as a separate function is to make
     * the code more organized.
     */
    set_lights();
    /* Once the lights are set, we can specify the points and faces that we
     * want drawn. We do all this in our 'draw_objects' helper function. See
     * the function for more details.
     *
     * The reason we have this procedure as a separate function is to make
     * the code more organized.
     */
    draw_objects();

    /* The following line of code has OpenGL do what is known as "double
     * buffering".
     *
     * Imagine this: You have a relatively slow computer that is telling OpenGL
     * to render something to display in the program window. Because your
     * computer is slow, OpenGL ends up rendering only part of the scene before
     * it displays it in the program window. The rest of the scene shows up a
     * second later. This effect is referred to as "flickering". You have most
     * likely experienced this sometime in your life when using a computer. It
     * is not the most visually appealing experience, right?
     *
     * To avoid the above situation, we need to tell OpenGL to display the
     * entire scene at once rather than rendering the scene one pixel at a
     * time. We do so by enabling "double buffering".
     *
     * Basically, double buffering is a technique where rendering is done using
     * two pixel grids of RGB values. One pixel grid is designated as the
     * "active buffer" while the other is designated as the "off-screen buffer".
     * Rendering is done on the off-screen buffer while the active buffer is
     * being displayed. Once the scene is fully rendered on the off-screen buffer,
     * the two buffers switch places so that the off-screen buffer becomes the
     * new active buffer and gets displayed while the old active buffer becomes
     * the new off-screen buffer. This process allows scenes to be fully rendered
     * onto the screen at once, avoiding the flickering effect.
     *
     * We actually enable double buffering in the 'main' function with the line:

       glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);

     * ^ 'GLUT_DOUBLE' tells OpenGL to use double buffering. The other two
     * parameters, 'GLUT_RGB' and 'GLUT_DEPTH', tell OpenGL to initialize the
     * RGB pixel grids and the depth buffer respectively.
     *
     * The following function, 'glutSwapBuffers', tells OpenGL to swap the
     * active and off-screen buffers.
     */
    glutSwapBuffers();
}


/* 'input_camera' function:
 *
 * This function inputs the details of the camera from the provided scene description file.
 *
 */
void input_camera(ifstream& fin) {
    string header;
    fin >> header;
    assert(header == "camera:");

    fin >> header; assert(header == "position");
    fin >> cam_position[0] >> cam_position[1] >> cam_position[2];

    fin >> header; assert(header == "orientation");
    fin >> cam_orientation_axis[0] >> cam_orientation_axis[1] >> cam_orientation_axis[2] >> cam_orientation_angle;
    cam_orientation_angle = rad2deg(cam_orientation_angle);

    fin >> header; assert(header == "near");
    fin >> near_param;

    fin >> header; assert(header == "far");
    fin >> far_param;

    fin >> header; assert(header == "left");
    fin >> left_param;

    fin >> header; assert(header == "right");
    fin >> right_param;

    fin >> header; assert(header == "top");
    fin >> top_param;

    fin >> header; assert(header == "bottom");
    fin >> bottom_param;
}

/* 'input' function:
 *
 * This function inputs all of the information from the scene description file.
 *
 */
void input(char* filename) {
    ifstream fin(filename, ifstream::in);
    input_camera(fin);
    string file_path (dirname(filename));

    string header;
    while (fin >> header) {
        if (header != "light") break;
        Point_Light new_light;
        fin >> new_light.position[0] >> new_light.position[1] >> new_light.position[2];
        new_light.position[3] = 1;
        string seperator; fin >> seperator;
        assert(seperator == ",");
        fin >> new_light.color[0] >> new_light.color[1] >> new_light.color[2];
        fin >> seperator;
        assert(seperator == ",");
        fin >> new_light.attenuation_k;

        lights.push_back(new_light);
    }

    assert(header == "objects:");
    string mesh_name;
    while (fin >> mesh_name) {
        if (mesh_name_to_mesh.find(mesh_name) != mesh_name_to_mesh.end()) {
            // This mesh name is already in the map -- we've reached the transform listing
            break;
        }
        string mesh_file_name;
        fin >> mesh_file_name;
        ifstream mesh_file((file_path + "/" + mesh_file_name).c_str(), ifstream::in);

        string vertex_or_normal_or_face;
        Mesh_Data mesh;

        int num_vertices = 0;
        mesh.vertices = new vector<Vertex*>();
        mesh.faces = new vector<Face*>();
        mesh.vertices->push_back(NULL); // Faces are given as 1-indexed, so make those indices correct.

        // Read in the model's vertices and faces.
        while (mesh_file >> vertex_or_normal_or_face) {
            if (vertex_or_normal_or_face == "v") {
                float first_number, second_number, third_number;
                mesh_file >> first_number >> second_number >> third_number;
                Vertex *v = new Vertex;
                v->x = first_number;
                v->y = second_number;
                v->z = third_number;
                mesh.vertices->push_back(v);
            }
            else if (vertex_or_normal_or_face == "f") {
                // Each line looks like f <v1> <v2> <v3>
                int v1, v2, v3;
                mesh_file >> v1 >> v2 >> v3;

                Face *f = new Face;
                f->idx1 = v1;
                f->idx2 = v2;
                f->idx3 = v3;
                mesh.faces->push_back(f);
            }
        }

        // Insert the mesh into the map
        mesh_name_to_mesh.insert(mesh_name_to_mesh.begin(), make_pair(mesh_name, mesh));
    }

    while (true) {
        // Create objects with transformations

        Object curr_object;
        curr_object.mesh = &(mesh_name_to_mesh[mesh_name]);

        curr_object.hevs = new vector<HEV*>();
        curr_object.hefs = new vector<HEF*>();

        build_HE(curr_object.mesh, curr_object.hevs, curr_object.hefs);
        for(int i = 1; i<curr_object.hevs->size(); ++i) {
            calc_vertex_normal(curr_object.hevs->at(i));
        }

        for(int i = 0; i<curr_object.mesh->faces->size(); ++i) {
            Face *curr_face = curr_object.mesh->faces->at(i);
            curr_object.vertex_buffer.push_back(*curr_object.mesh->vertices->at(curr_face->idx1));
            curr_object.vertex_buffer.push_back(*curr_object.mesh->vertices->at(curr_face->idx2));
            curr_object.vertex_buffer.push_back(*curr_object.mesh->vertices->at(curr_face->idx3));
            // Normals generated from HE mesh
            curr_object.normal_buffer.push_back(curr_object.hevs->at(curr_face->idx1)->normal);
            curr_object.normal_buffer.push_back(curr_object.hevs->at(curr_face->idx2)->normal);
            curr_object.normal_buffer.push_back(curr_object.hevs->at(curr_face->idx3)->normal);
        }

        fin >> header; assert(header == "ambient");
        fin >> curr_object.ambient_reflect[0] >> curr_object.ambient_reflect[1] >> curr_object.ambient_reflect[2];
        fin >> header; assert(header == "diffuse");
        fin >> curr_object.diffuse_reflect[0] >> curr_object.diffuse_reflect[1] >> curr_object.diffuse_reflect[2];
        fin >> header; assert(header == "specular");
        fin >> curr_object.specular_reflect[0] >> curr_object.specular_reflect[1] >> curr_object.specular_reflect[2];
        fin >> header; assert(header == "shininess");
        fin >> curr_object.shininess;

        string translation_rotation_or_scale;
        double x_component, y_component, z_component;

        while (fin >> translation_rotation_or_scale) {
            Transform t;
            if (translation_rotation_or_scale == "t") {
                fin >> t.components[0] >> t.components[1] >> t.components[2];
                t.rotation_angle = 0;
                t.transform_type = 0;
            }
            else if (translation_rotation_or_scale == "r") {
                fin >> t.components[0] >> t.components[1] >> t.components[2];
                fin >> t.rotation_angle;
                t.rotation_angle = rad2deg(t.rotation_angle);
                t.transform_type = 1;
            }
            else if (translation_rotation_or_scale == "s") {
                fin >> t.components[0] >> t.components[1] >> t.components[2];
                t.rotation_angle = 0;
                t.transform_type = 2;
            }
            else {
                // We've hit the next object

                mesh_name = translation_rotation_or_scale;
                break;
            }

            curr_object.transforms.push_back(t);
        }

        objects.push_back(curr_object);

        if (fin.eof()) {
            break;
        }
    }
}


/* 'init_lights' function:
 *
 * This function has OpenGL enable its built-in lights to represent our point
 * lights.
 *
 * OpenGL has 8 built-in lights in all, each one with its own unique, integer
 * ID value. When setting the properties of a light, we need to tell OpenGL
 * the ID value of the light we are modifying.
 *
 * The first light's ID value is stored in 'GL_LIGHT0'. The second light's ID
 * value is stored in 'GL_LIGHT1'. And so on. The eighth and last light's ID
 * value is stored in 'GL_LIGHT7'.
 *
 * The properties of the lights are set using the 'glLightfv' and 'glLightf'
 * functions as you will see below.
 */
void init_lights()
{
    /* The following line of code tells OpenGL to enable lighting calculations
     * during its rendering process. This tells it to automatically apply the
     * Phong reflection model or lighting model to every pixel it will render.
     */
    glEnable(GL_LIGHTING);

    int num_lights = lights.size();

    for(int i = 0; i < num_lights; ++i)
    {
        /* In this loop, we are going to associate each of our point lights
         * with one of OpenGL's built-in lights. The simplest way to do this
         * is to just let our first point light correspond to 'GL_LIGHT0', our
         * second point light correspond to 'GL_LIGHT1', and so on. i.e. let:
         *
         * 'lights[0]' have an ID value of 'GL_LIGHT0'
         * 'lights[1]' have an ID value of 'GL_LIGHT1'
         * etc...
         */
        int light_id = GL_LIGHT0 + i;

        glEnable(light_id);

        /* The following lines of code use 'glLightfv' to set the color of
         * the light. The parameters for 'glLightfv' are:
         *
         * - enum light_ID: an integer between 'GL_LIGHT0' and 'GL_LIGHT7'
         * - enum property: this varies depending on what you are setting
         *                  e.g. 'GL_AMBIENT' for the light's ambient component
         * - float* values: a set of values to set for the specified property
         *                  e.g. an array of RGB values for the light's color
         *
         * OpenGL actually lets us specify different colors for the ambient,
         * diffuse, and specular components of the light. However, since we
         * are used to only working with one overall light color, we will
         * just set every component to the light color.
         */
        glLightfv(light_id, GL_AMBIENT, lights[i].color);
        glLightfv(light_id, GL_DIFFUSE, lights[i].color);
        glLightfv(light_id, GL_SPECULAR, lights[i].color);

        /* The following line of code sets the attenuation k constant of the
         * light. The difference between 'glLightf' and 'glLightfv' is that
         * 'glLightf' is used for when the parameter is only one value like
         * the attenuation constant while 'glLightfv' is used for when the
         * parameter is a set of values like a color array. i.e. the third
         * parameter of 'glLightf' is just a float instead of a float*.
         */
        glLightf(light_id, GL_QUADRATIC_ATTENUATION, lights[i].attenuation_k);
    }
}

/* 'set_lights' function:
 *
 * While the 'init_lights' function enables and sets the colors of the lights,
 * the 'set_lights' function is supposed to position the lights.
 *
 * You might be wondering why we do not just set the positions of the lights in
 * the 'init_lights' function in addition to the other properties. The reason
 * for this is because OpenGL does lighting computations after it applies the
 * Modelview Matrix to points. This means that the lighting computations are
 * effectively done in camera space. Hence, to ensure that we get the correct
 * lighting computations, we need to make sure that we position the lights
 * correctly in camera space.
 *
 * Now, the 'glLightfv' function, when used to position a light, applies all
 * the current Modelview Matrix to the given light position. This means that
 * to correctly position lights in camera space, we should call the 'glLightfv'
 * function to position them AFTER the Modelview Matrix has been modified by
 * the necessary camera transformations. As you can see in the 'display'
 * function, this is exactly what we do.
 */
void set_lights()
{
    int num_lights = lights.size();

    for(int i = 0; i < num_lights; ++i)
    {
        int light_id = GL_LIGHT0 + i;

        glLightfv(light_id, GL_POSITION, lights[i].position);
    }
}

/* 'draw_objects' function:
 *
 * This function has OpenGL render our objects to the display screen. It
 */
void draw_objects()
{
    int num_objects = objects.size();

    glMultMatrixd(get_current_rotation());

    for(int i = 0; i < num_objects; ++i)
    {
        /* The current Modelview Matrix is actually stored at the top of a
         * stack in OpenGL. The following function, 'glPushMatrix', pushes
         * another copy of the current Modelview Matrix onto the top of the
         * stack. This results in the top two matrices on the stack both being
         * the current Modelview Matrix. Let us call the copy on top 'M1' and
         * the copy that is below it 'M2'.
         *
         * The reason we want to use 'glPushMatrix' is because we need to
         * modify the Modelview Matrix differently for each object we need to
         * render, since each object is affected by different transformations.
         * We use 'glPushMatrix' to essentially keep a copy of the Modelview
         * Matrix before it is modified by an object's transformations. This
         * copy is our 'M2'. We then modify 'M1' and use it to render the
         * object. After we finish rendering the object, we will pop 'M1' off
         * the stack with the 'glPopMatrix' function so that 'M2' returns to
         * the top of the stack. This way, we have the old unmodified Modelview
         * Matrix back to edit for the next object we want to render.
         */
        glPushMatrix();
        /* The following brace is not necessary, but it keeps things organized.
         */
        {
            int num_transforms = objects[i].transforms.size();

            /* The loop tells OpenGL to modify our modelview matrix with the
             * desired geometric transformations for this object.
             * We make the calls in the REVERSE order of how the transformations are specified
             * because OpenGL edits our modelview matrix using post-multiplication (see above
             * at the notes regarding the camera transforms in display()).
             */
            for(int j = num_transforms-1; j >= 0; --j)
            {
                if(objects[i].transforms[j].transform_type == 0) {
                    glTranslatef(objects[i].transforms[j].components[0],
                                 objects[i].transforms[j].components[1],
                                 objects[i].transforms[j].components[2]);
                }
                else if(objects[i].transforms[j].transform_type == 1) {
                    glRotatef(objects[i].transforms[j].rotation_angle,
                              objects[i].transforms[j].components[0],
                              objects[i].transforms[j].components[1],
                              objects[i].transforms[j].components[2]);
                }
                else if(objects[i].transforms[j].transform_type == 2) {
                    glScalef(objects[i].transforms[j].components[0],
                             objects[i].transforms[j].components[1],
                             objects[i].transforms[j].components[2]);
                }
            }

            /* The 'glMaterialfv' and 'glMaterialf' functions tell OpenGL
             * the material properties of the surface we want to render.
             * The parameters for 'glMaterialfv' are (in the following order):
             *
             * - enum face: Options are 'GL_FRONT' for front-face rendering,
             *              'GL_BACK' for back-face rendering, and
             *              'GL_FRONT_AND_BACK' for rendering both sides.
             * - enum property: this varies on what you are setting up
             *                  e.g. 'GL_AMBIENT' for ambient reflectance
             * - float* values: a set of values for the specified property
             *                  e.g. an array of RGB values for the reflectance
             *
             * The 'glMaterialf' function is the same, except the third
             * parameter is only a single float value instead of an array of
             * values. 'glMaterialf' is used to set the shininess property.
             */
            glMaterialfv(GL_FRONT, GL_AMBIENT, objects[i].ambient_reflect);
            glMaterialfv(GL_FRONT, GL_DIFFUSE, objects[i].diffuse_reflect);
            glMaterialfv(GL_FRONT, GL_SPECULAR, objects[i].specular_reflect);
            glMaterialf(GL_FRONT, GL_SHININESS, objects[i].shininess);

            /* The next few lines of code are how we tell OpenGL to render
             * geometry for us. First, let us look at the 'glVertexPointer'
             * function.
             *
             * 'glVertexPointer' tells OpenGL the specifications for our
             * "vertex array". As a recap of the comments from the 'Object'
             * struct, the "vertex array" stores all the faces of the surface
             * we want to render. The faces are stored in the array as
             * consecutive points. For instance, if our surface were a cube,
             * then our "vertex array" could be the following:
             *
             * [face1vertex1, face1vertex2, face1vertex3, face1vertex4,
             *  face2vertex1, face2vertex2, face2vertex3, face2vertex4,
             *  face3vertex1, face3vertex2, face3vertex3, face3vertex4,
             *  face4vertex1, face4vertex2, face4vertex3, face4vertex4,
             *  face5vertex1, face5vertex2, face5vertex3, face5vertex4,
             *  face6vertex1, face6vertex2, face6vertex3, face6vertex4]
             *
             * Obviously to us, some of the vertices in the array are repeats.
             * However, the repeats cannot be avoided since OpenGL requires
             * this explicit specification of the faces.
             *
             * The parameters to the 'glVertexPointer' function are as
             * follows:
             *
             * - int num_points_per_face: this is the parameter that tells
             *                            OpenGL where the breaks between
             *                            faces are in the vertex array.
             *                            Below, we set this parameter to 3,
             *                            which tells OpenGL to treat every
             *                            set of 3 consecutive vertices in
             *                            the vertex array as 1 face. So
             *                            here, our vertex array is an array
             *                            of triangle faces.
             *                            If we were using the example vertex
             *                            array above, we would have set this
             *                            parameter to 4 instead of 3.
             * - enum type_of_coordinates: this parameter tells OpenGL whether
             *                             our vertex coordinates are ints,
             *                             floats, doubles, etc. In our case,
             *                             we are using floats, hence 'GL_FLOAT'.
             * - sizei stride: this parameter specifies the number of bytes
             *                 between consecutive vertices in the array.
             *                 Most often, you will set this parameter to 0
             *                 (i.e. no offset between consecutive vertices).
             * - void* pointer_to_array: this parameter is the pointer to
             *                           our vertex array.
             */
            glVertexPointer(3, GL_FLOAT, 0, &objects[i].vertex_buffer[0]);
            /* The "normal array" is the equivalent array for normals.
             * Each normal in the normal array corresponds to the vertex
             * of the same index in the vertex array.
             *
             * The 'glNormalPointer' function has the following parameters:
             *
             * - enum type_of_normals: e.g. int, float, double, etc
             * - sizei stride: same as the stride parameter in 'glVertexPointer'
             * - void* pointer_to_array: the pointer to the normal array
             */
            glNormalPointer(GL_FLOAT, 0, &objects[i].normal_buffer[0]);

            int buffer_size = objects[i].vertex_buffer.size();

            glDrawArrays(GL_TRIANGLES, 0, buffer_size);
        }
        /* As discussed before, we use 'glPopMatrix' to get back the
         * version of the Modelview Matrix that we had before we specified
         * the object transformations above. We then move on in our loop
         * to the next object we want to render.
         */
        glPopMatrix();
    }

}

/* 'mouse_pressed' function:
 *
 * This function is meant to respond to mouse clicks and releases. The
 * parameters are:
 *
 * - int button: the button on the mouse that got clicked or released,
 *               represented by an enum
 * - int state: either 'GLUT_DOWN' or 'GLUT_UP' for specifying whether the
 *              button was pressed down or released up respectively
 * - int x: the x screen coordinate of where the mouse was clicked or released
 * - int y: the y screen coordinate of where the mouse was clicked or released
 *
 * The function doesn't really do too much besides set some variables that
 * we need for the 'mouse_moved' function.
 */
void mouse_pressed(int button, int state, int x, int y)
{
    /* If the left-mouse button was clicked down, then...
     */
    if(button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
    {
        /* Store the mouse position in our global variables.
         */
        mouse_x_start = x;
        mouse_y_start = y;

        /* Since the mouse is being pressed down, we set our 'is_pressed"
         * boolean indicator to true.
         */
        is_pressed = true;
    }
    /* If the left-mouse button was released up, then...
     */
    else if(button == GLUT_LEFT_BUTTON && state == GLUT_UP)
    {
        /* Lock in the current rotation matrix.
         */
        last_rotation = product(last_rotation, curr_rotation);
        curr_rotation = identity();

        /* Mouse is no longer being pressed, so set our indicator to false.
         */
        is_pressed = false;
    }
}

/* 'mouse_moved' function:
 *
 * This function is meant to respond to when the mouse is being moved. There
 * are just two parameters to this function:
 *
 * - int x: the x screen coordinate of where the mouse was clicked or released
 * - int y: the y screen coordinate of where the mouse was clicked or released
 *
 * We compute our camera rotation angles based on the mouse movement in this
 * function.
 */
void mouse_moved(int x, int y)
{
    /* If the left-mouse button is being clicked down...
     */
    if(is_pressed)
    {
        mouse_x_current = x;
        mouse_y_current = y;
        curr_rotation = compute_rotation_quaternion(mouse_x_current, mouse_y_current, mouse_x_start, mouse_y_start);

        /* Tell OpenGL that it needs to re-render our scene with the new model rotation.
         */
        glutPostRedisplay();
    }
}

/* 'deg2rad' function:
 *
 * Converts given angle in degrees to radians.
 */
float deg2rad(float angle)
{
    return angle * M_PI / 180.0;
}

/* 'rad2deg' function:
 *
 * Converts given angle in radians to degress.
 */
float rad2deg(float angle)
{
    return angle * 180.0 / M_PI;
}


/* key_pressed function:
 *
 * Smooth if space is pressed.
 */
void key_pressed(unsigned char key, int x, int y) {
    if(key == ' ') {
        for(int i = 0; i<objects.size(); ++i) {
            smooth(&objects[i]);
        }
        glutPostRedisplay();
    }
}


/* The 'main' function:
 *
 * This function is short, but is basically where everything comes together.
 */
int main(int argc, char* argv[])
{
    assert(argc == 5);
    xres = atoi(argv[2]);
    yres = atoi(argv[3]);
    h = atof(argv[4]);

    /* 'glutInit' intializes the GLUT (Graphics Library Utility Toolkit) library.
     * This is necessary, since a lot of the functions we used above and below
     * are from the GLUT library.
     *
     * 'glutInit' takes the 'main' function arguments as parameters. This is not
     * too important for us, but it is possible to give command line specifications
     * to 'glutInit' by putting them with the 'main' function arguments.
     */
    glutInit(&argc, argv);
    /* The following line of code tells OpenGL that we need a double buffer,
     * a RGB pixel buffer, and a depth buffer.
     */
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    /* The following line tells OpenGL to create a program window of size
     * 'xres' by 'yres'.
     */
    glutInitWindowSize(xres, yres);
    /* The following line tells OpenGL to set the program window in the top-left
     * corner of the computer screen (0, 0).
     */
    glutInitWindowPosition(0, 0);
    /* The following line tells OpenGL to name the program window "Test".
     */
    glutCreateWindow("Test");

    /* Call our 'init' function...
     */
    init(argv[1]);
    /* Specify to OpenGL our display function.
     */
    glutDisplayFunc(display);
    /* Specify to OpenGL our reshape function.
     */
    glutReshapeFunc(reshape);
    /* Specify to OpenGL our function for handling mouse presses.
     */
    glutMouseFunc(mouse_pressed);
    /* Specify to OpenGL our function for handling mouse movement.
     */
    glutMotionFunc(mouse_moved);
    /* Specify to OpenGL our function for handling key presses.
     */
    glutKeyboardFunc(key_pressed);
    /* The following line tells OpenGL to start the "event processing loop". This
     * is an infinite loop where OpenGL will continuously use our display, reshape,
     * mouse, and keyboard functions to essentially run our program.
     */
    glutMainLoop();
}
