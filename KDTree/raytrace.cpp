// Philip Carr
// CS 171 Assignment 5
// November 27, 2018
// smoooth.cpp

#define GL_GLEXT_PROTOTYPES 1
#ifdef MAC
#include <GLUT/glut.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#endif


#include <math.h>
#define _USE_MATH_DEFINES

#include <iostream>
#include <vector>

#include "scene_read.h"
#include "quaternion.h"

#include "PNGMaker.hpp"

using namespace std;

////////////////////////////////////////////////////////////////////////////////
// quaternion class member function definitions.

// Default constructor for a quaternion.
quaternion::quaternion() {
    s = 0;
    v[0] = 0;
    v[1] = 0;
    v[2] = 0;
}

// 4-argument constructor for a quaternion.
quaternion::quaternion(const float s, const float v0, const float v1,
                       const float v2) {
    this->s = s;
    this->v[0] = v0;
    this->v[1] = v1;
    this->v[2] = v2;
}

// Return the quaternion's s-value.
float quaternion::get_s() const {
    return this->s;
}

// Return the quaternion's v-vector.
float *quaternion::get_v() {
    return this->v;
}

// Set the value of s of a quaternion.
void quaternion::set_s(const float s) {
    this->s = s;
}

// Set the vector v of a quaternion.
void quaternion::set_v(const float v0, const float v1, const float v2) {
    this->v[0] = v0;
    this->v[1] = v1;
    this->v[2] = v2;
}

// Add a quaternion q to the given quaternion.
void quaternion::add(quaternion &q) {
    float qs = q.get_s();
    float *qv = q.get_v();
    float qx = qv[0];
    float qy = qv[1];
    float qz = qv[2];

    this->s += qs;
    this->v[0] += qx;
    this->v[1] += qy;
    this->v[2] += qz;
}

// Multiply a scalar s to the given quaternion.
void quaternion::s_mult(const float s) {
    this->s *= s;
    this->v[0] *= s;
    this->v[1] *= s;
    this->v[2] *= s;
}

// Return the product of the quaternion q and the given quaternion.
quaternion quaternion::q_mult(quaternion &q) const { // Use Eigen
    float qs = q.get_s();
    float *qv = q.get_v();
    float qx = qv[0];
    float qy = qv[1];
    float qz = qv[2];

    float mult_s = this->s * qs - this->v[0] * qx - this->v[1] * qy
                   - this->v[2] * qz;
    float mult_v0 = this->s * qx + this->v[0] * qs + this->v[1] * qz
                    - this->v[2] * qy;
    float mult_v1 = this->s * qy + this->v[1] * qs + this->v[2] * qx
                    - this->v[0] * qz;
    float mult_v2 = this->s * qz + this->v[2] * qs + this->v[0] * qy
                    - this->v[1] * qx;

    return quaternion(mult_s, mult_v0, mult_v1, mult_v2);
}

// Return the complex conjugate of the given quaternion.
quaternion quaternion::conj() const {
    float neg_v0 = -this->v[0];
    float neg_v1 = -this->v[1];
    float neg_v2 = -this->v[2];
    return quaternion(this->s, neg_v0, neg_v1, neg_v2);
}

// Return the norm of the given quaternion (sqrt(s^2 + x^2 + y^2 + z^2)).
float quaternion::norm() const {
    return sqrt(this->s * this->s + this->v[0] * this->v[0]
           + this->v[1] * this->v[1] + this->v[2] * this->v[2]);
}

// Return the inverse of the given quaternion (q^*/(|q|^2)).
quaternion quaternion::inv() const {
    float norm = this->norm();
    quaternion q_star_normalized = this->conj();
    q_star_normalized.set_s(q_star_normalized.get_s() / (norm * norm));
    float *v = q_star_normalized.get_v();
    q_star_normalized.set_v(v[0] / (norm * norm), v[1] / (norm * norm),
                            v[2] / (norm * norm));
    return q_star_normalized;
}

// Normalize the given quaternion.
void quaternion::normalize() {
    float norm = this->norm();
    this->s = this->s / norm;
    this->v[0] = this->v[0] / norm;
    this->v[1] = this->v[1] / norm;
    this->v[2] = this->v[2] / norm;
}

////////////////////////////////////////////////////////////////////////////////

void init(void);
void readShaders();
void reshape(int width, int height);
void display(void);

void init_lights();
void set_lights();
void draw_objects();

void mouse_pressed(int button, int state, int x, int y);
void mouse_moved(int x, int y);
void key_pressed(unsigned char key, int x, int y);

void update_rotations(int x, int y);

void transform_objects(bool init);

// ray tracing function helpers
bool is_in_shadow(Vector3f p, Vector3f l);

////////////////////////////////////////////////////////////////////////////////

// Camera and lists of lights and objects.

Camera cam;
vector<Point_Light> lights;
vector<Object> objects;
vector<Object> original_objects;

Vector3f cam_pos_3f;
Vector3f cam_dir_3f;

////////////////////////////////////////////////////////////////////////////////

/* The following are parameters for creating an interactive first-person camera
 * view of the scene. The variables will make more sense when explained in
 * context, so you should just look at the 'mousePressed', 'mouseMoved', and
 * 'keyPressed' functions for the details.
 */

int mouse_x, mouse_y;
float mouse_scale_x, mouse_scale_y;

const float step_size = 0.2;
const float x_view_step = 90.0, y_view_step = 90.0;
float x_view_angle = 0, y_view_angle = 0;

quaternion last_rotation;
quaternion current_rotation;

bool is_pressed = false;
bool wireframe_mode = false;

int xres;
int yres;

GLenum shaderProgram;
string vertProgFileName, fragProgFileName;
GLint n_lights;

int mode;

float h;
int smooth_toggle = false;
bool smooth_turned_off = false;

////////////////////////////////////////////////////////////////////////////////

/* Initialze OpenGL system and organize scene data (camera, lights, and
 * objects).
 */
void init(string filename, int mode) {
    lights = read_lights(filename);

    /* Check mode value here: if mode == 0, use Gouraud Shading. If mode == 1,
     * use Phong shaders with the readShaders function.
     */
    if (mode == 0) {
        glShadeModel(GL_SMOOTH);
    }
    else {
        readShaders();
    }

    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);

    glEnable(GL_DEPTH_TEST);

    glEnable(GL_NORMALIZE);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);


    glMatrixMode(GL_PROJECTION);

    glLoadIdentity();

    cam = read_camera(filename);
    // cam.pos[0] = 0;
    // cam.pos[1] = 0;
    // cam.pos[2] = 3;
    cam.fov = 60.0;
    cam.aspect = (float) xres / yres;

    // float rx = cam.ori_axis[0];
    // float ry = cam.ori_axis[1];
    // float rz = cam.ori_axis[2];
    // float angle = cam.ori_angle * M_PI / 180.0;
    // float rotation_axis_magnitude = rx * rx + ry * ry + rz * rz;
    // rx /= rotation_axis_magnitude;
    // ry /= rotation_axis_magnitude;
    // rz /= rotation_axis_magnitude;
    // float a00 = rx * rx + (1 - rx * rx) * cos(angle);
    // float a01 = rx * ry * (1 - cos(angle)) - rz * sin(angle);
    // float a02 = rx * rz * (1 - cos(angle)) + ry * sin(angle);
    // float a10 = ry * rx * (1 - cos(angle)) + rz * sin(angle);
    // float a11 = ry * ry + (1 - ry * ry) * cos(angle);
    // float a12 = ry * rz * (1 - cos(angle)) - rx * sin(angle);
    // float a20 = rz * rx * (1 - cos(angle)) - ry * sin(angle);
    // float a21 = rz * ry * (1 - cos(angle)) + rx * sin(angle);
    // float a22 = rz * rz + (1 - rz * rz) * cos(angle);
    // Matrix4f cam_rotation << a00, a01, a02, 0,
    //     a10, a11, a12, 0,
    //     a20, a21, a22, 0,
    //     0, 0, 0, 1;

    glFrustum(cam.l, cam.r, cam.b, cam.t, cam.n, cam.f);

    glMatrixMode(GL_MODELVIEW);

    objects = read_objects(filename);
    original_objects = read_objects(filename);

    init_lights();

    last_rotation = quaternion(1, 0, 0, 0);
    current_rotation = quaternion(1, 0, 0, 0);

    transform_objects(true);
}

/**
 * Read glsl vertex shader and fragment shader files and compile them together
 * to create a shader program.
 */
void readShaders() {
   string vertProgramSource, fragProgramSource;

   ifstream vertProgFile(vertProgFileName.c_str());
   if (! vertProgFile)
      cerr << "Error opening vertex shader program\n";
   ifstream fragProgFile(fragProgFileName.c_str());
   if (! fragProgFile)
      cerr << "Error opening fragment shader program\n";

   getline(vertProgFile, vertProgramSource, '\0');
   const char* vertShaderSource = vertProgramSource.c_str();

   getline(fragProgFile, fragProgramSource, '\0');
   const char* fragShaderSource = fragProgramSource.c_str();

   char buf[1024];
   GLsizei blah;

   // Initialize shaders
   GLenum vertShader, fragShader;

   shaderProgram = glCreateProgram();

   vertShader = glCreateShader(GL_VERTEX_SHADER);
   glShaderSource(vertShader, 1, &vertShaderSource, NULL);
   glCompileShader(vertShader);

   GLint isCompiled = 0;
   glGetShaderiv(vertShader, GL_COMPILE_STATUS, &isCompiled);
   if(isCompiled == GL_FALSE)
   {
      GLint maxLength = 0;
      glGetShaderiv(vertShader, GL_INFO_LOG_LENGTH, &maxLength);

      // The maxLength includes the NULL character
      std::vector<GLchar> errorLog(maxLength);
      glGetShaderInfoLog(vertShader, maxLength, &maxLength, &errorLog[0]);

      // Provide the infolog in whatever manor you deem best.
      // Exit with failure.
      for (int i = 0; i < errorLog.size(); i++)
         cout << errorLog[i];
      glDeleteShader(vertShader); // Don't leak the shader.
      return;
   }

   fragShader = glCreateShader(GL_FRAGMENT_SHADER);
   glShaderSource(fragShader, 1, &fragShaderSource, NULL);
   glCompileShader(fragShader);

   isCompiled = 0;
   glGetShaderiv(fragShader, GL_COMPILE_STATUS, &isCompiled);
   if(isCompiled == GL_FALSE)
   {
      GLint maxLength = 0;
      glGetShaderiv(fragShader, GL_INFO_LOG_LENGTH, &maxLength);

      // The maxLength includes the NULL character
      std::vector<GLchar> errorLog(maxLength);
      glGetShaderInfoLog(fragShader, maxLength, &maxLength, &errorLog[0]);

      // Provide the infolog in whatever manor you deem best.
      // Exit with failure.
      for (int i = 0; i < errorLog.size(); i++)
         cout << errorLog[i];
      glDeleteShader(fragShader); // Don't leak the shader.
      return;
   }

   glAttachShader(shaderProgram, vertShader);
   glAttachShader(shaderProgram, fragShader);
   glLinkProgram(shaderProgram);
   cerr << "Enabling fragment program: " << gluErrorString(glGetError()) << endl;
   glGetProgramInfoLog(shaderProgram, 1024, &blah, buf);
   cerr << buf;

   cerr << "Enabling program object" << endl;
   glUseProgram(shaderProgram);

   // Pass the total number of lights into the shader program.
   n_lights = glGetUniformLocation(shaderProgram, "n_lights");
   glUniform1i(n_lights, (int) lights.size());
}

/**
 * Reshape the image whenever the window size changes.
 */
void reshape(int width, int height) {
    height = (height == 0) ? 1 : height;
    width = (width == 0) ? 1 : width;

    glViewport(0, 0, width, height);

    mouse_scale_x = (float) (cam.r - cam.l) / (float) width;
    mouse_scale_y = (float) (cam.t - cam.b) / (float) height;

    glutPostRedisplay();
}

/**
 * Display the scene using OpenGL.
 */
void display(void) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glLoadIdentity();

    float ori_mag = sqrt(cam.ori_axis[0] * cam.ori_axis[0]
                         + cam.ori_axis[1] * cam.ori_axis[1]
                         + cam.ori_axis[2] * cam.ori_axis[2]);
    // cout << cam.ori_angle << endl;
    // glRotatef(cam.ori_angle * 180 / M_PI,
    //            cam.ori_axis[0] / ori_mag, cam.ori_axis[1] / ori_mag,
    //            cam.ori_axis[2] / ori_mag);
    glRotatef(cam.ori_angle, // camera angle in given in degrees
              cam.ori_axis[0] / ori_mag, cam.ori_axis[1] / ori_mag,
              cam.ori_axis[2] / ori_mag);

    glTranslatef(-cam.pos[0], -cam.pos[1], -cam.pos[2]);

    quaternion rotation = current_rotation.q_mult(last_rotation);
    float qs = rotation.get_s();
    float *v = rotation.get_v();
    float qx = v[0];
    float qy = v[1];
    float qz = v[2];
    float rotation_matrix[16] = {
        1 - 2 * qy * qy - 2 * qz * qz, 2 * (qx * qy + qz * qs),
        2 * (qx * qz - qy * qs), 0,

        2 * (qx * qy - qz * qs), 1 - 2 * qx * qx - 2 * qz * qz,
        2 * (qy * qz + qx * qs), 0,

        2 * (qx * qz + qy * qs), 2 * (qy * qz - qx * qs),
        1 - 2 * qx * qx - 2 * qy * qy, 0,

        0, 0, 0, 1
    };

    glMultMatrixf(rotation_matrix);

    Matrix4f q_v_transform;
    q_v_transform <<
         rotation_matrix[0], rotation_matrix[1],
         rotation_matrix[2], rotation_matrix[3],

         rotation_matrix[4], rotation_matrix[5],
         rotation_matrix[6], rotation_matrix[7],

         rotation_matrix[8], rotation_matrix[9],
         rotation_matrix[10], rotation_matrix[11],

         rotation_matrix[12], rotation_matrix[13],
         rotation_matrix[14], rotation_matrix[15];

    Matrix4f q_v_transform_inv = q_v_transform.inverse();
    Matrix4f q_n_transform = q_v_transform_inv.transpose();

    Vector4f cam_pos_4f;
    cam_pos_4f << cam_pos_3f(0), cam_pos_3f(0), cam_pos_3f(0), 1.0;
    cam_pos_4f = q_v_transform * cam_pos_4f;
    cam_pos_3f(0) = cam_pos_4f(0) / cam_pos_4f(3);
    cam_pos_3f(1) = cam_pos_4f(1) / cam_pos_4f(3);
    cam_pos_3f(2) = cam_pos_4f(2) / cam_pos_4f(3);

    Vector4f cam_dir_4f;
    cam_dir_4f << cam_dir_3f(0), cam_dir_3f(0), cam_dir_3f(0), 1.0;
    cam_dir_4f = q_v_transform * cam_dir_4f;
    cam_dir_3f(0) = cam_dir_4f(0) / cam_dir_4f(3);
    cam_dir_3f(1) = cam_dir_4f(1) / cam_dir_4f(3);
    cam_dir_3f(2) = cam_dir_4f(2) / cam_dir_4f(3);

    // for (int i = 0; i < (int) objects.size(); i++) {
    //     // Transform all the vertices in the given vector of vertices.
    //     vector<Vertex> v_vector = objects[i].transformed_vertices;
    //
    //     vector<Vertex> t_vertices;
    //     vector<Vec3f> t_normals;
    //
    //     for (int k = 0; k < (int) v_vector.size(); k++) {
    //         Vector4f v;
    //         v << v_vector[k].x, v_vector[k].y, v_vector[k].z, 1.0;
    //         Vector4f tv;
    //         tv = q_v_transform * v;
    //         Vertex t_vertex;
    //         t_vertex.x = tv(0) / tv(3);
    //         t_vertex.y = tv(1) / tv(3);
    //         t_vertex.z = tv(2) / tv(3);
    //         t_vertices.push_back(t_vertex);
    //     }
    //
    //     vector<Vec3f> vn_vector = objects[i].transformed_normals;
    //     for (int k = 0; k < (int) vn_vector.size(); k++) {
    //         Vector4f vn;
    //         vn << vn_vector[k].x, vn_vector[k].y, vn_vector[k].z, 1.0;
    //         Vector4f tvn;
    //         tvn = q_n_transform * vn;
    //         Vec3f t_vnorm;
    //         t_vnorm.x = tvn(0) / tvn(3);
    //         t_vnorm.y = tvn(1) / tvn(3);
    //         t_vnorm.z = tvn(2) / tvn(3);
    //         t_normals.push_back(t_vnorm);
    //     }
    //
    //     objects[i].transformed_vertices = t_vertices;
    //     objects[i].transformed_normals = t_normals;
    // }

    set_lights();

    draw_objects();

    glutSwapBuffers();
}

/**
 * Initialze the lights of the scene.
 */
void init_lights() {

    glEnable(GL_LIGHTING);

    int num_lights = lights.size();

    for(int i = 0; i < num_lights; ++i)
    {

        int light_id = GL_LIGHT0 + i;

        glEnable(light_id);

        glLightfv(light_id, GL_AMBIENT, lights[i].color);
        glLightfv(light_id, GL_DIFFUSE, lights[i].color);
        glLightfv(light_id, GL_SPECULAR, lights[i].color);

        glLightf(light_id, GL_QUADRATIC_ATTENUATION, lights[i].k);
    }
}

/**
 * Set the lights of the scene.
 */
void set_lights() {
    int num_lights = lights.size();

    for(int i = 0; i < num_lights; ++i)
    {
        int light_id = GL_LIGHT0 + i;

        glLightfv(light_id, GL_POSITION, lights[i].pos);
    }
}

/**
 * Transform the objects into world space in advance, so the ray tracing
 * function can have access to the world space coordinates of the objects.
 * Additional note: maybe quaternion rotation should be applied here after
 * object transformations are applied here.
 */
void transform_objects(bool init) {
    int num_objects = objects.size();
    for (int i = 0; i < num_objects; i++) {
        if (init) {
            int num_transform_sets = objects[i].transform_sets.size();
            for (int j = num_transform_sets - 1; j >= 0 ; j--) {
                Obj_Transform transform = objects[i].transform_sets[j];
                Matrix4f new_matrix;
                if (transform.type == "t") {
                    new_matrix << 1, 0, 0, transform.components[0],
                                  0, 1, 0, transform.components[1],
                                  0, 0, 1, transform.components[2],
                                  0, 0, 0, 1;
                    objects[i].t_transform_matrices.push_back(new_matrix);
                }
                else if (transform.type == "r") {
                    float rx = transform.components[0];
                    float ry = transform.components[1];
                    float rz = transform.components[2];
                    float angle = transform.rotation_angle;
                    float rotation_axis_magnitude = rx * rx + ry * ry + rz * rz;
                    rx /= rotation_axis_magnitude;
                    ry /= rotation_axis_magnitude;
                    rz /= rotation_axis_magnitude;
                    float a00 = rx * rx + (1 - rx * rx) * cos(angle);
                    float a01 = rx * ry * (1 - cos(angle)) - rz * sin(angle);
                    float a02 = rx * rz * (1 - cos(angle)) + ry * sin(angle);
                    float a10 = ry * rx * (1 - cos(angle)) + rz * sin(angle);
                    float a11 = ry * ry + (1 - ry * ry) * cos(angle);
                    float a12 = ry * rz * (1 - cos(angle)) - rx * sin(angle);
                    float a20 = rz * rx * (1 - cos(angle)) - ry * sin(angle);
                    float a21 = rz * ry * (1 - cos(angle)) + rx * sin(angle);
                    float a22 = rz * rz + (1 - rz * rz) * cos(angle);
                    new_matrix << a00, a01, a02, 0,
                                  a10, a11, a12, 0,
                                  a20, a21, a22, 0,
                                  0, 0, 0, 1;
                    objects[i].t_transform_matrices.push_back(new_matrix);
                    objects[i].n_transform_matrices.push_back(new_matrix);
                }
                else {
                    glScalef(transform.components[0], transform.components[1],
                             transform.components[2]);
                    new_matrix << transform.components[0], 0, 0, 0,
                                  0, transform.components[1], 0, 0,
                                  0, 0, transform.components[2], 0,
                                  0, 0, 0, 1;
                    objects[i].t_transform_matrices.push_back(new_matrix);
                    objects[i].n_transform_matrices.push_back(new_matrix);
                }
            }
        }

        Matrix4f v_transform;
        v_transform << 1, 0, 0, 0,
                       0, 1, 0, 0,
                       0, 0, 1, 0,
                       0, 0, 0, 1;

        Matrix4f n_transform;
        n_transform << 1, 0, 0, 0,
                       0, 1, 0, 0,
                       0, 0, 1, 0,
                       0, 0, 0, 1;

        // Multiply all the transformations together into one transformation.
        vector<Matrix4f> tm_vector = objects[i].t_transform_matrices;
        for (int i = 0; i < (int) tm_vector.size(); i++) {
            v_transform = tm_vector[i] * v_transform;
        }

        vector<Matrix4f> nm_vector = objects[i].n_transform_matrices;
        for (int i = 0; i < (int) nm_vector.size(); i++) {
            n_transform = nm_vector[i] * n_transform;
        }

        Matrix4f n_transform_inv = n_transform.inverse();
        Matrix4f vn_transform = n_transform_inv.transpose();

        // Transform all the vertices in the given vector of vertices.
        vector<Vertex> v_vector = objects[i].vertex_buffer;

        vector<Vertex> t_vertices;
        vector<Vec3f> t_normals;

        for (int k = 0; k < (int) v_vector.size(); k++) {
            Vector4f v;
            v << v_vector[k].x, v_vector[k].y, v_vector[k].z, 1.0;
            Vector4f tv;
            tv = v_transform * v;
            Vertex t_vertex;
            t_vertex.x = tv(0) / tv(3);
            t_vertex.y = tv(1) / tv(3);
            t_vertex.z = tv(2) / tv(3);
            t_vertices.push_back(t_vertex);
        }

        vector<Vec3f> vn_vector = objects[i].normal_buffer;
        for (int k = 0; k < (int) vn_vector.size(); k++) {
            Vector4f vn;
            vn << vn_vector[k].x, vn_vector[k].y, vn_vector[k].z, 1.0;
            Vector4f tvn;
            tvn = vn_transform * vn;
            Vec3f t_vnorm;
            t_vnorm.x = tvn(0) / tvn(3);
            t_vnorm.y = tvn(1) / tvn(3);
            t_vnorm.z = tvn(2) / tvn(3);
            t_normals.push_back(t_vnorm);
        }

        objects[i].transformed_vertices = t_vertices;
        objects[i].transformed_normals = t_normals;
    }
}

/**
 * Draw the objects in the scene. New note: add functionality to add array of
 * all triangle faces of all objects for ray tracing function to find ray-
 * object intersections.
 */
void draw_objects() {
    int num_objects = objects.size();

    for(int i = 0; i < num_objects; ++i)
    {
        // glPushMatrix();

        {
            // int num_transform_sets = objects[i].transform_sets.size();

            /* Modify the current modelview matrix with the
             * geometric transformations for this object.
             */
            // for(int j = num_transform_sets - 1; j >= 0 ; j--) {
            //     Obj_Transform transform = objects[i].transform_sets[j];
            //     if (transform.type == "t") {
            //         glTranslatef(transform.components[0],
            //                      transform.components[1],
            //                      transform.components[2]);
            //     }
            //     else if (transform.type == "r") {
            //         glRotatef(transform.rotation_angle,
            //                   transform.components[0], transform.components[1],
            //                   transform.components[2]);
            //     }
            //     else {
            //         glScalef(transform.components[0], transform.components[1],
            //                  transform.components[2]);
            //     }
            // }

            glMaterialfv(GL_FRONT, GL_AMBIENT, objects[i].ambient_reflect);
            glMaterialfv(GL_FRONT, GL_DIFFUSE, objects[i].diffuse_reflect);
            glMaterialfv(GL_FRONT, GL_SPECULAR, objects[i].specular_reflect);
            glMaterialf(GL_FRONT, GL_SHININESS, objects[i].shininess);

            // glVertexPointer(3, GL_FLOAT, 0, &objects[i].vertex_buffer[0]);
            glVertexPointer(3, GL_FLOAT, 0, &objects[i].transformed_vertices[0]);

            // glNormalPointer(GL_FLOAT, 0, &objects[i].normal_buffer[0]);
            glNormalPointer(GL_FLOAT, 0, &objects[i].transformed_normals[0]);

            int buffer_size = objects[i].vertex_buffer.size();

            if(!wireframe_mode)
                glDrawArrays(GL_TRIANGLES, 0, buffer_size);
            else
                for(int j = 0; j < buffer_size; j += 3)
                    glDrawArrays(GL_LINE_LOOP, j, 3);
        }

        // glPopMatrix();
    }
}

/**
 * Function to tell OpenGL what to do when the (left) mouse (button) is pressed.
 */
void mouse_pressed(int button, int state, int x, int y) {
    if(button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
    {
        mouse_x = x;
        mouse_y = y;

        is_pressed = true;
    }
    else if(button == GLUT_LEFT_BUTTON && state == GLUT_UP)
    {
        is_pressed = false;

        last_rotation = current_rotation.q_mult(last_rotation);
        current_rotation = quaternion(1, 0, 0, 0);
    }
}

/**
 * Update the quaternion rotations current_rotation and last_rotation using the
 * Arcball algorithm.
 */
void update_rotations(int x, int y) {
    float x1 = (mouse_x - ((float (xres - 1)) / 2)) / ((float) xres / 2);
    float y1 = -(mouse_y - ((float (yres - 1)) / 2)) / ((float) yres / 2);
    float z1;
    if (x1 * x1 + y1 * y1 <= 1) {
        z1 = sqrt(1 - x1 * x1 - y1 * y1);
    }
    else {
        z1 = (float) 0;
    }

    float x2 = (x - ((float (xres - 1)) / 2)) / ((float) xres / 2);
    float y2 = -(y - ((float (yres - 1)) / 2)) / ((float) yres / 2);
    float z2;
    if (x2 * x2 + y2 * y2 <= 1) {
        z2 = sqrt(1 - x2 * x2 - y2 * y2);
    }
    else {
        z2 = (float) 0;
    }

    float v1_norm = sqrt(x1 * x1 + y1 * y1 + z1 * z1);
    float v2_norm = sqrt(x2 * x2 + y2 * y2 + z2 * z2);

    float theta = acos(min((float) 1, (x1 * x2 + y1 * y2 + z1 * z2)
                                      / (v1_norm * v2_norm)));

    float u[3] = {y1 * z2 - z1 * y2, z1 * x2 - x1 * z2, x1 * y2 - y1 * x2};
    float u_magnitude = sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    if (u_magnitude != 0) {
        u[0] = u[0] / u_magnitude;
        u[1] = u[1] / u_magnitude;
        u[2] = u[2] / u_magnitude;
    }

    float qs = cos(theta / 2);
    float qx = u[0] * sin(theta / 2);
    float qy = u[1] * sin(theta / 2);
    float qz = u[2] * sin(theta / 2);

    current_rotation = quaternion(qs, qx, qy, qz);
    current_rotation.normalize();

    last_rotation = current_rotation.q_mult(last_rotation);
}

/**
 * Tells OpenGL what to do when the mouse is moved.
 */
void mouse_moved(int x, int y) {
    if(is_pressed)
    {
        update_rotations(x, y);

        mouse_x = x;
        mouse_y = y;

        glutPostRedisplay();
    }
}

/* 'deg2rad' function:
 *
 * Converts given angle in degrees to radians.
 */
float deg2rad(float angle) {
    return angle * M_PI / 180.0;
}

// Ray tracing code below ------------------------------------------------------

struct Ray {
    // Boolean to tell if ray is actually represents an intersection.
    bool is_intersect;

    float origin_x;
    float origin_y;
    float origin_z;

    float direction_x;
    float direction_y;
    float direction_z;

    Vector3f getLocation(float t) {
        Vector3f loc;
        loc << origin_x + t * direction_x,
            origin_y + t * direction_y,
            origin_z + t * direction_z;
        return loc;
    }
};

struct Intersection {
    Ray intersection_ray;
    int obj_num;
};

enum SplitDir { x, y, z };

struct Tri {
    Vertex v1;
    Vertex v2;
    Vertex v3;
};

typedef struct KDTree {
    // Tree structure
    struct KDTree* left;
    struct KDTree* right;

    // Each level of the tree has the same splitting direction, which alternates
    // x -> y -> z -> x -> ...
    // for the purpose of making the tree well-behaved I guess
    SplitDir dir;

    // Axis-aligned bounding box (AABB)
    float xmin, xmax;
    float ymin, ymax;
    float zmin, zmax;

    // Faces in AABB (only for leaf nodes)
    vector<Tri> tris;
    vector<Tri> normals;
} KDTree;

float centroid(Tri f, SplitDir dir) {
    switch (dir) {
        case x:
            return (f.v1.x + f.v2.x + f.v3.x) / 3.0;
        case y:
            return (f.v1.y + f.v2.y + f.v3.y) / 3.0;
        case z:
            return (f.v1.z + f.v2.z + f.v3.z) / 3.0;
    }
}

void constructKDTree(KDTree* root, vector<Tri> trisInTree, vector<Tri> normalsInTree) {
    root->tris = trisInTree;
    root->normals = normalsInTree;
    if(trisInTree.size() <= 15) {
        return; // Arbitrary stopping point; maybe a tree depth is better?
    }
    SplitDir dir = root->dir;
    // Find median of tris in split direction (using centroid of tris)
    float split = 0.0f;
    // Approx median as average of centroids of triangles (can do an actual median-finding algorithm later if need be)
    for (int i = 0; i<trisInTree.size(); ++i) {
        //root->tris.push_back(trisInTree[i]);
        //root->normals.push_back(normalsInTree[i]);
        float centr = centroid(trisInTree[i], dir);
        split += centr;
    }
    split /= (1.0 * trisInTree.size());


    // Partition tris based on median split
    vector<Tri> leftTris, rightTris;
    vector<Tri> leftNormals, rightNormals;
    for (int i = 0; i<trisInTree.size(); ++i) {
        //float centr = centroid(trisInTree[i], dir);
        Tri tri = trisInTree[i];
        double tSplit1, tSplit2, tSplit3;
        switch (dir) {
            case x:
                tSplit1 = tri.v1.x;
                tSplit2 = tri.v2.x;
                tSplit3 = tri.v3.x;
                break;
            case y:
                tSplit1 = tri.v1.y;
                tSplit2 = tri.v2.y;
                tSplit3 = tri.v3.y;
                break;
            case z:
                tSplit1 = tri.v1.z;
                tSplit2 = tri.v2.z;
                tSplit3 = tri.v3.z;
                break;
        }
        if (tSplit1 < split || tSplit2 < split || tSplit3 < split) {
            leftTris.push_back(trisInTree[i]);
            leftNormals.push_back(normalsInTree[i]);
        }
        if (tSplit1 > split || tSplit2 > split || tSplit3 > split) {
            rightTris.push_back(trisInTree[i]);
            rightNormals.push_back(normalsInTree[i]);
        }
    }

    // Recurse
    SplitDir nextDir;
    switch (dir) {
        case x:
            nextDir = y;
            break;
        case y:
            nextDir = z;
            break;
        case z:
            nextDir = x;
            break;
    }

    root->left = NULL;
    root->right = NULL;
    root->dir = dir;

    if (!leftTris.empty()) {
        root->left = new KDTree();
        root->left->dir = nextDir;
        root->left->xmin = root->xmin;
        root->left->xmax = root->xmax;
        root->left->ymin = root->ymin;
        root->left->ymax = root->ymax;
        root->left->zmin = root->zmin;
        root->left->zmax = root->zmax;
        switch (dir) {
            case x:
                root->left->xmax = split;
                break;
            case y:
                root->left->ymax = split;
                break;
            case z:
                root->left->zmax = split;
                break;
        }
        constructKDTree(root->left, leftTris, leftNormals);
    }
    if (!rightTris.empty()) {
        root->right = new KDTree();
        root->right->dir = nextDir;
        root->right->xmin = root->xmin;
        root->right->xmax = root->xmax;
        root->right->ymin = root->ymin;
        root->right->ymax = root->ymax;
        root->right->zmin = root->zmin;
        root->right->zmax = root->zmax;
        switch (dir) {
            case x:
                root->right->xmin = split;
                break;
            case y:
                root->right->ymin = split;
                break;
            case z:
                root->right->zmin = split;
                break;
        }
        constructKDTree(root->right, rightTris, rightNormals);
    }
}

void destructKDTree(KDTree* tree) {
    KDTree* left = tree->left;
    KDTree* right = tree->right;

    delete(tree);
    // Recurse
    if (left != NULL)
        destructKDTree(left);
    if (right != NULL)
        destructKDTree(right);
}

bool doesIntersect(KDTree* box, Ray ray) {
    double tmin = (box->xmin - ray.origin_x) / ray.direction_x;
    double tmax = (box->xmax - ray.origin_x) / ray.direction_x;
    if (tmin > tmax) swap(tmin, tmax);

    double tymin = (box->ymin - ray.origin_y) / ray.direction_y;
    double tymax = (box->ymax - ray.origin_y) / ray.direction_y;
    if (tymin > tymax) swap(tymin, tymax);

    if (tmin > tymax || tmax < tymin) return false;

    tmin = max(tmin, tymin);
    tmax = min(tmax, tymax);

    double tzmin = (box->zmin - ray.origin_z) / ray.direction_z;
    double tzmax = (box->zmax - ray.origin_z) / ray.direction_z;
    if (tzmin > tzmax) swap(tzmin, tzmax);

    if (tmin > tzmax || tmax < tzmin) return false;
    return true;
}

Intersection closestTreeIntersection(KDTree* tree, Ray ray, int i) {
    Intersection noHit;
    noHit.intersection_ray.is_intersect = false;
    noHit.obj_num = -1;
    if (!doesIntersect(tree, ray)) return noHit;

    if (tree->left != NULL || tree->right != NULL) {
        if (tree->left == NULL) {
            return closestTreeIntersection(tree->right, ray, i);
        }
        if (tree->right == NULL) {
            return closestTreeIntersection(tree->left, ray, i);
        }
        Intersection leftIntersection = closestTreeIntersection(tree->left, ray, i);
        Intersection rightIntersection = closestTreeIntersection(tree->right, ray, i);

        // If either is a noHit, return the other.
        if (leftIntersection.obj_num == -1) return rightIntersection;
        if (rightIntersection.obj_num == -1) return leftIntersection;

        // Return the closest of leftIntersection and rightIntersection, since both are valid.
        float iray_x = leftIntersection.intersection_ray.origin_x;
        float iray_y = leftIntersection.intersection_ray.origin_y;
        float iray_z = leftIntersection.intersection_ray.origin_z;
        double distanceLeft = sqrt(pow(iray_x - ray.origin_x, 2)
                              + pow(iray_y - ray.origin_y, 2)
                              + pow(iray_z - ray.origin_z, 2));
        iray_x = rightIntersection.intersection_ray.origin_x;
        iray_y = rightIntersection.intersection_ray.origin_y;
        iray_z = rightIntersection.intersection_ray.origin_z;
        double distanceRight = sqrt(pow(iray_x - ray.origin_x, 2)
                              + pow(iray_y - ray.origin_y, 2)
                              + pow(iray_z - ray.origin_z, 2));
        if (distanceLeft < distanceRight) return leftIntersection;
        return rightIntersection;
    }

    // Otherwise, we've hit a leaf node.  Look through all of
    // the contents of the tree and see which the closest is,
    // if any.
    Intersection closestIntersection = noHit;
    double closestIntersectionDist = 999999999;
    for (int j = 0; j < (int) tree->tris.size(); ++j) {
        //cout << j << endl;
        Intersection intersection;

        float xp = ray.origin_x;
        float yp = ray.origin_y;
        float zp = ray.origin_z;

        float xd = ray.direction_x;
        float yd = ray.direction_y;
        float zd = ray.direction_z;

        float xa = tree->tris[j].v1.x;
        float ya = tree->tris[j].v1.y;
        float za = tree->tris[j].v1.z;

        float xb = tree->tris[j].v2.x;
        float yb = tree->tris[j].v2.y;
        float zb = tree->tris[j].v2.z;

        float xc = tree->tris[j].v3.x;
        float yc = tree->tris[j].v3.y;
        float zc = tree->tris[j].v3.z;

        Matrix3f A;
        A << xa - xb, xa - xc, xd,
             ya - yb, ya - yc, yd,
             za - zb, za - zc, zd;

        Matrix3f A_inv = A.inverse();

        Vector3f b;
        b << xa - xp, ya - yp, za - zp;

        Vector3f soln;
        soln = A_inv * b;

        float beta = soln(0);
        float gamma = soln(1);
        float t = soln(2);

        if (beta > 0 && gamma > 0 && beta + gamma < 1) {
            //cout << " " << xa << " " << ya << " " << za << endl;
            //cout << " " << xb << " " << yb << " " << zb << endl;
            //cout << " " << xc << " " << yc << " " << zc << endl;
            float alpha = 1 - beta - gamma;
            Ray intersection_ray;
            intersection_ray.origin_x =
                ray.origin_x + t * ray.direction_x;
            intersection_ray.origin_y =
                ray.origin_y + t * ray.direction_y;
            intersection_ray.origin_z =
                ray.origin_z + t * ray.direction_z;

            intersection_ray.direction_x =
                alpha * tree->normals[j].v1.x
                + beta * tree->normals[j].v2.x
                + gamma * tree->normals[j].v3.x;
            intersection_ray.direction_y =
                alpha * tree->normals[j].v1.y
                + beta * tree->normals[j].v2.y
                + gamma * tree->normals[j].v3.y;
            intersection_ray.direction_z =
                alpha * tree->normals[j].v1.z
                + beta * tree->normals[j].v2.z
                + gamma * tree->normals[j].v3.z;

            /*
            double normalization = intersection_ray.direction_x * intersection_ray.direction_x + intersection_ray.direction_y * intersection_ray.direction_y + intersection_ray.direction_z * intersection_ray.direction_z;
            intersection_ray.direction_x /= normalization;
            intersection_ray.direction_y /= normalization;
            intersection_ray.direction_z /= normalization;
            */

            intersection_ray.is_intersect = true;

            intersection.intersection_ray = intersection_ray;
            intersection.obj_num = i;

            // Is this intersection the closest to the ray origin so far?
            double dist = sqrt(pow(intersection_ray.origin_x - ray.origin_x, 2)
                             + pow(intersection_ray.origin_y - ray.origin_y, 2)
                             + pow(intersection_ray.origin_z - ray.origin_z, 2));

            //cout << dist << endl;
            //cout << intersection_ray.origin_x << " " << intersection_ray.origin_y << " " << intersection_ray.origin_z << endl;
            if (dist < closestIntersectionDist) {
                closestIntersection = intersection;
                closestIntersectionDist = dist;
            }
        }
    }

    return closestIntersection;
}

/**
 * Return a vector of all the intersection structs of the objects in the scene.
 */
Intersection findClosestIntersection(vector<KDTree*> &trees, Ray camera_ray) {
    Intersection noHit;
    noHit.intersection_ray.is_intersect = false;
    noHit.obj_num = -1;
    Intersection closestIntersection = noHit;
    double closestIntersectionDist = 999999999;
    for (int i = 0; i < (int) objects.size(); i++) {
        // cout << (int) objects[i].transformed_vertices.size() << endl;
        // cout << (int) objects[i].transformed_normals.size() << endl;

        Intersection inter = closestTreeIntersection(trees[i], camera_ray, i);

        double dist = sqrt(pow(inter.intersection_ray.origin_x - camera_ray.origin_x, 2)
                         + pow(inter.intersection_ray.origin_y - camera_ray.origin_y, 2)
                         + pow(inter.intersection_ray.origin_z - camera_ray.origin_z, 2));

        if (dist < closestIntersectionDist) {
            closestIntersection = inter;
            closestIntersectionDist = dist;
        }

        /*
        for (int j = 0; j < (int) objects[i].transformed_vertices.size();
             j+=3) {
            Intersection intersection;

            float xp = camera_ray.origin_x;
            float yp = camera_ray.origin_y;
            float zp = camera_ray.origin_z;

            float xd = camera_ray.direction_x;
            float yd = camera_ray.direction_y;
            float zd = camera_ray.direction_z;

            float xa = objects[i].transformed_vertices[j].x;
            float ya = objects[i].transformed_vertices[j].y;
            float za = objects[i].transformed_vertices[j].z;

            float xb = objects[i].transformed_vertices[j+1].x;
            float yb = objects[i].transformed_vertices[j+1].y;
            float zb = objects[i].transformed_vertices[j+1].z;

            float xc = objects[i].transformed_vertices[j+2].x;
            float yc = objects[i].transformed_vertices[j+2].y;
            float zc = objects[i].transformed_vertices[j+2].z;

            Matrix3f A;
            A << xa - xb, xa - xc, xd,
                 ya - yb, ya - yc, yd,
                 za - zb, za - zc, zd;

            Matrix3f A_inv = A.inverse();

            Vector3f b;
            b << xa - xp, ya - yp, za - zp;

            Vector3f soln;
            soln = A_inv * b;

            float beta = soln(0);
            float gamma = soln(1);
            float t = soln(2);

            if (beta > 0 && gamma > 0 && beta + gamma < 1) {
                // cout << "Intersection found at face\n"
                //      << "  (" << xa << ", " << ya << ", " << za << "),\n"
                //      << "  (" << xb << ", " << yb << ", " << zb << "),\n"
                //      << "  (" << xc << ", " << yc << ", " << zc << ")\n"
                //      << "  with t = " << t << "\n" << endl;

                float alpha = 1 - beta - gamma;
                Ray intersection_ray;
                intersection_ray.origin_x =
                    camera_ray.origin_x + t * camera_ray.direction_x;
                intersection_ray.origin_y =
                    camera_ray.origin_y + t * camera_ray.direction_y;
                intersection_ray.origin_z =
                    camera_ray.origin_z + t * camera_ray.direction_z;

                intersection_ray.direction_x =
                    alpha * objects[i].transformed_normals[j].x
                    + beta * objects[i].transformed_normals[j+1].x
                    + gamma * objects[i].transformed_normals[j+2].x;
                intersection_ray.direction_y =
                    alpha * objects[i].transformed_normals[j].y
                    + beta * objects[i].transformed_normals[j+1].y
                    + gamma * objects[i].transformed_normals[j+2].y;
                intersection_ray.direction_z =
                    alpha * objects[i].transformed_normals[j].z
                    + beta * objects[i].transformed_normals[j+1].z
                    + gamma * objects[i].transformed_normals[j+2].z;

                intersection_ray.is_intersect = true;

                intersection.intersection_ray = intersection_ray;
                intersection.obj_num = i;

                intersections.push_back(intersection);
                // Vector3f point;
                // point << intersection_ray.origin_x, intersection_ray.origin_y,
                //          intersection_ray.origin_z;
                // Vector3f light_pos;
                // light_pos << lights[0].pos[0], lights[0].pos[1],
                //              lights[0].pos[2];
                // cout << "Intersection point in shadow of light 0: "
                //      << is_in_shadow(point, light_pos); // infinite loops...
            }
        }
    */
    }
    return closestIntersection;
}

/**
 * Return true if the current point on the surface of the corresponding
 * surface is in the shadow of another surface. This function determines
 * whether an intersection occurs along the ray from the light to the point
 * p.
 */
bool is_in_shadow(Vector3f p, Vector3f l, vector<KDTree*> trees) {
    return false;
    Ray shadow_ray;
    shadow_ray.origin_x = l(0);
    shadow_ray.origin_y = l(1);
    shadow_ray.origin_z = l(2);

    Vector3f a_vec = p - l;
    shadow_ray.direction_x = a_vec(0);
    shadow_ray.direction_y = a_vec(1);
    shadow_ray.direction_z = a_vec(2);

    // Find all intersection rays of the primitives.
    //vector<Intersection> intersections;
    Intersection intersection = findClosestIntersection(trees, shadow_ray);

    if (!intersection.intersection_ray.is_intersect) return false;

    float light_to_point_distance = a_vec.norm();
    /*
    Intersection closest_intersection;

    // Gets rid of numerical errors leading to random dark spots appearing
    // on the surfaces of objects that are directly illuminated.
    float distance_threshold = 0.9999;

    for (int i = 0; i < (int) intersections.size(); i++) {
        Intersection intersection = intersections[i];
        Ray intersection_ray = intersection.intersection_ray;

        // Check if the current intersection ray is the closest to the
        // camera
        float iray_x = intersection_ray.origin_x;
        float iray_y = intersection_ray.origin_y;
        float iray_z = intersection_ray.origin_z;

        float distance = sqrt(pow(iray_x - shadow_ray.origin_x, 2)
                              + pow(iray_y - shadow_ray.origin_y, 2)
                              + pow(iray_z - shadow_ray.origin_z, 2));

        if (distance < distance_threshold * light_to_point_distance
            && intersection_ray.is_intersect) {
            return true;
        }
    }
    return false;
    */
    Ray intersection_ray = intersection.intersection_ray;

    // Check if the current intersection ray is the closest to the
    // camera
    float iray_x = intersection_ray.origin_x;
    float iray_y = intersection_ray.origin_y;
    float iray_z = intersection_ray.origin_z;

    float distance = sqrt(pow(iray_x - shadow_ray.origin_x, 2)
                          + pow(iray_y - shadow_ray.origin_y, 2)
                          + pow(iray_z - shadow_ray.origin_z, 2));
    float distance_threshold = 0.9999;
    /*
    cout << distance << endl;
    if(distance < distance_threshold * light_to_point_distance) {cout << "is_in_shadow " << intersection.obj_num << endl;}
    else {cout << "is_not_in_shadow" << endl;}
    */
    return distance < distance_threshold * light_to_point_distance;
}

/**
 * Return the color of a given vertex with a given vertex normal vector
 * determined by the Phong lighting model.
 */
Vector3f lighting(
    const Vector3f v, const Vector3f vn,
    const Object &obj,
    const vector<Point_Light> &lights,
    const Vector3f cam_pos,
    const vector<KDTree*> &trees) {

    Vector3f vn_v = vn.normalized();

    Vector3f cd;
    cd << obj.diffuse_reflect[0], obj.diffuse_reflect[1],
          obj.diffuse_reflect[2];
    Vector3f ca;
    ca << obj.ambient_reflect[0], obj.ambient_reflect[1],
          obj.ambient_reflect[2];
    Vector3f cs;
    cs << obj.specular_reflect[0], obj.specular_reflect[1],
          obj.specular_reflect[2];
    float s = obj.shininess;

    Vector3f diffuse_sum;
    Vector3f specular_sum;

    diffuse_sum << 0, 0, 0;
    specular_sum << 0, 0, 0;

    Vector3f cam_v_vec = cam_pos - v;
    Vector3f cam_dir = cam_v_vec.normalized();

    /* Multiplier for attenuation to make the scene brighter (since the
     * value below is less than 1).
     */
    float k_multiplier = 0.05;

    for (int i = 0; i < (int) lights.size(); i++) {
        Point_Light l = lights[i];
        Vector3f lp;
        Vector3f lc;
        Vector3f l_dir;

        lp << l.pos[0], l.pos[1], l.pos[2];

        /* Check if the current point is in the shadow of another object.
         * If so, the diffuse and specular components of the current light's
         * illumination are not added.
         */
        if (is_in_shadow(v, lp, trees)) {
            continue;
        }

        lc << l.color[0], l.color[1], l.color[2];
        l_dir << lp - v;
        float dSquared = l_dir.squaredNorm();
        lc /= (1 + k_multiplier * l.k * dSquared);
        l_dir = l_dir.normalized();

        Vector3f l_diff = lc * max((float) 0.0, (float) vn_v.dot(l_dir));
        diffuse_sum += l_diff;

        Vector3f dir_sum = cam_dir + l_dir;
        Vector3f l_spec =
            lc * pow(max((float) 0.0, (float)
                                      vn_v.dot(dir_sum.normalized())), s);
        specular_sum += l_spec;
    }

    Vector3f c;
    c(0) = min((float) 1.0, ca(0) + diffuse_sum(0) * cd(0)
                            + specular_sum(0) * cs(0));
    c(1) = min((float) 1.0, ca(1) + diffuse_sum(1) * cd(1)
                            + specular_sum(1) * cs(1));
    c(2) = min((float) 1.0, ca(2) + diffuse_sum(2) * cd(2)
                            + specular_sum(2) * cs(2));
    return c;
}

/* Ray traces the scene. */
void raytrace() {
    PNGMaker png = PNGMaker(xres, yres);

    Vector3f e1;
    Vector3f e2;
    Vector3f e3;

    Vector4f e1_4d;
    Vector4f e2_4d;
    Vector4f e3_4d;
    e1_4d << 0, 0, -1, 1;
    e2_4d << 1, 0, 0, 1;
    e3_4d << 0, 1, 0, 1;

    // Rotate e1_4d by camera rotation transformation.
    Matrix4f camera_rotation;
    float rx = cam.ori_axis[0];
    float ry = cam.ori_axis[1];
    float rz = cam.ori_axis[2];
    float angle = cam.ori_angle;

    float rotation_axis_magnitude = sqrt(rx * rx + ry * ry + rz * rz);
    rx /= rotation_axis_magnitude;
    ry /= rotation_axis_magnitude;
    rz /= rotation_axis_magnitude;

    float a00 = rx * rx + (1 - rx * rx) * cos(angle);
    float a01 = rx * ry * (1 - cos(angle)) - rz * sin(angle);
    float a02 = rx * rz * (1 - cos(angle)) + ry * sin(angle);
    float a10 = ry * rx * (1 - cos(angle)) + rz * sin(angle);
    float a11 = ry * ry + (1 - ry * ry) * cos(angle);
    float a12 = ry * rz * (1 - cos(angle)) - rx * sin(angle);
    float a20 = rz * rx * (1 - cos(angle)) - ry * sin(angle);
    float a21 = rz * ry * (1 - cos(angle)) + rx * sin(angle);
    float a22 = rz * rz + (1 - rz * rz) * cos(angle);

    camera_rotation <<
        a00, a01, a02, 0,
        a10, a11, a12, 0,
        a20, a21, a22, 0,
        0, 0, 0, 1;

    e1_4d = camera_rotation * e1_4d;
    e2_4d = camera_rotation * e2_4d;
    e3_4d = camera_rotation * e3_4d;

    e1 << e1_4d(0), e1_4d(1), e1_4d(2);
    e2 << e2_4d(0), e2_4d(1), e2_4d(2);
    e3 << e3_4d(0), e3_4d(1), e3_4d(2);

    Ray camera_ray;
    camera_ray.origin_x = cam.pos[0];
    camera_ray.origin_y = cam.pos[1];
    camera_ray.origin_z = cam.pos[2];

    Vector3f cam_pos;
    cam_pos << camera_ray.origin_x, camera_ray.origin_y,
               camera_ray.origin_z;

    float percentage_increase = 10;
    float percentage_print = percentage_increase;
    cout << "Starting ray tracing" << endl;

    // Create kd-trees for all objects
    vector<KDTree*> trees;
    for(int i = 0; i<(int)objects.size(); ++i) {
        vector<Tri> triangles, normals;
        for (int j = 0; j < (int) objects[i].transformed_vertices.size(); j+=3) {
            Tri t;
            t.v1 = objects[i].transformed_vertices[j];
            t.v2 = objects[i].transformed_vertices[j+1];
            t.v3 = objects[i].transformed_vertices[j+2];
            triangles.push_back(t);
            Tri n;
            n.v1.x = objects[i].transformed_normals[j].x;
            n.v1.y = objects[i].transformed_normals[j].y;
            n.v1.z = objects[i].transformed_normals[j].z;
            n.v2.x = objects[i].transformed_normals[j+1].x;
            n.v2.y = objects[i].transformed_normals[j+1].y;
            n.v2.z = objects[i].transformed_normals[j+1].z;
            n.v3.x = objects[i].transformed_normals[j+2].x;
            n.v3.y = objects[i].transformed_normals[j+2].y;
            n.v3.z = objects[i].transformed_normals[j+2].z;
            normals.push_back(n);
        }
        KDTree* tree = new KDTree;
        tree->left = NULL;
        tree->right = NULL;
        tree->dir = x;
        tree->xmin = -1000000;
        tree->xmax = 1000000;
        tree->ymin = -1000000;
        tree->ymax = 1000000;
        tree->zmin = -1000000;
        tree->zmax = 1000000;
        constructKDTree(tree, triangles, normals);
        trees.push_back(tree);
    }

    for (int i = 0; i < xres; i++) {
        for (int j = 0; j < yres; j++) {
            if ((float) 100 * i / xres >= percentage_print) {
                cout << (float) 100 * i / xres << '%' << " done raytracing"
                     << endl;
                percentage_print += percentage_increase;
            }
            //cout << "Ray tracing pixel (" << i << ", " << j << ")" << endl;
            Vector3f color;
            color << 0.0, 0.0, 0.0;

            // Determine the current camera direction.
            float n = cam.n;
            float fov = cam.fov * M_PI / 180.0;
            float h = 2.0 * n * tan(fov / 2.0);
            float w = cam.aspect * h;

            float xi = (i - xres / 2) * (w / xres);
            float yj = (j - yres / 2) * (h / yres);
            Vector3f a_vec = cam.n * e1 + xi * e2 + yj * e3;
            camera_ray.direction_x = a_vec(0);
            camera_ray.direction_y = a_vec(1);
            camera_ray.direction_z = a_vec(2);

            // Find all intersection rays of the primitives.
            //vector<Intersection> intersections;
            // cout << "Finding intersections" << endl;
            Intersection intersection = findClosestIntersection(trees, camera_ray);
            // cout << "Found intersections" << endl;

            /*
            float min_distance = 1000000;
            Intersection closest_intersection;
            bool found_intersection = false;
            for (int k = 0; k < (int) intersections.size(); k++) {
                Intersection intersection = intersections[k];
                Ray intersection_ray = intersection.intersection_ray;

                // Check if the current intersection ray is the closest to
                // the camera.
                float iray_x = intersection_ray.origin_x;
                float iray_y = intersection_ray.origin_y;
                float iray_z = intersection_ray.origin_z;

                float distance =
                    sqrt(pow(iray_x - camera_ray.origin_x, 2)
                         + pow(iray_y - camera_ray.origin_y, 2)
                         + pow(iray_z - camera_ray.origin_z, 2));

                if (distance < min_distance
                    && intersection_ray.is_intersect) {
                    min_distance = distance;
                    closest_intersection = intersection;
                    found_intersection = true;
                }
            }
            */

            /* Use the Phong lighting model with shadowing for the closest
             * intersection point (if an intersection occurs).
             */
            if (intersection.intersection_ray.is_intersect) {
                //cout << intersection.intersection_ray.origin_x << " " << intersection.intersection_ray.origin_y << " " << intersection.intersection_ray.origin_z << endl;
                Vector3f v;
                Vector3f n;

                v << intersection.intersection_ray.origin_x,
                     intersection.intersection_ray.origin_y,
                     intersection.intersection_ray.origin_z;
                Vector3f endpoint =
                    intersection.intersection_ray.getLocation(1.0);
                n = endpoint - v;
                //cout << n.x() << " " << n.y() << " " << n.z() << endl;

                color = lighting(v, n, objects[intersection.obj_num],
                                 lights, cam_pos, trees);
            }

            png.setPixel(i, j, color(0), color(1), color(2));
        }
    }

    for(int i = 0; i<(int)objects.size(); ++i) {
        destructKDTree(trees[i]);
    }

    if (png.saveImage()) {
        fprintf(stderr, "Error: couldn't save PNG image\n");
    } else {
        printf("DONE!\n");
    }
}

// Ray tracing code above ------------------------------------------------------

/**
 * Function tells OpenGL what to do when a key is pressed.
 */
void key_pressed(unsigned char key, int x, int y) {

    if(key == 'q') {
        exit(0);
    }

    else if(key == 't') {
        wireframe_mode = !wireframe_mode;

        glutPostRedisplay();
    }
    /* If the 'h' key is pressed, toggle smoothing on or off depending on the
     * previous state of smooth_toggle.
     */
    else if(key == 'h') {
        smooth_toggle = !smooth_toggle;
        if (!smooth_toggle) {
            smooth_turned_off = true;
        }
    }
    /* If the 'u' key is pressed, increment the time step h by 0.0001 and revert
     * back to the original object.
     */
    else if (key == 'u') {
        if (smooth_toggle) {
            smooth_toggle = false;
            smooth_turned_off = true;
        }
        h += 0.0001;
        cout << "h incremented by 0.0001: h = " << h << endl;
    }
    /* If the 'i' key is pressed, decrement the time step h by 0.0001 and revert
     * back to the original object.
     */
    else if (key == 'i') {
        if (smooth_toggle) {
            smooth_toggle = false;
            smooth_turned_off = true;
        }
        h -= 0.0001;
        cout << "h decremented by 0.0001: h = " << h << endl;
    }
    /* If the 'o' key is pressed, double the time step h and revert back to the
     * original object.
     */
    else if (key == 'o') {
        if (smooth_toggle) {
            smooth_toggle = false;
            smooth_turned_off = true;
        }
        h *= 2;
        cout << "h doubled: h = " << h << endl;
    }
    /* If the 'p' key is pressed, halve the time step h and revert back to the
     * original object.
     */
    else if (key == 'p') {
        if (smooth_toggle) {
            smooth_toggle = false;
            smooth_turned_off = true;
        }
        h /= 2;
        cout << "h halved: h = " << h << endl;
    }
    /* If the 'j' key is pressed, set the time step h to 0 and revert back to
     * the original object.
     */
    else if (key == 'j') {
        if (smooth_toggle) {
            smooth_toggle = false;
            smooth_turned_off = true;
        }
        h = 0;
        cout << "set h = " << h << endl;
    }
    else if (key == 'r') {
        // Ray camera_ray;
        // camera_ray.origin_x = 0;
        // camera_ray.origin_y = 0;
        // camera_ray.origin_z = 3;
        // camera_ray.direction_x = 0;
        // camera_ray.direction_y = 0;
        // camera_ray.direction_z = -1;
        // vector<Intersection> intersections;
        // findAllIntersections(intersections, camera_ray);
        raytrace();
    }
    else {
        float x_view_rad = deg2rad(x_view_angle);

        /* 'w' for step forward
         */
        if(key == 'w')
        {
            cam.pos[0] += step_size * sin(x_view_rad);
            cam.pos[2] -= step_size * cos(x_view_rad);
            glutPostRedisplay();
        }
        /* 'a' for step left
         */
        else if(key == 'a')
        {
            cam.pos[0] -= step_size * cos(x_view_rad);
            cam.pos[2] -= step_size * sin(x_view_rad);
            glutPostRedisplay();
        }
        /* 's' for step backward
         */
        else if(key == 's')
        {
            cam.pos[0] -= step_size * sin(x_view_rad);
            cam.pos[2] += step_size * cos(x_view_rad);
            glutPostRedisplay();
        }
        /* 'd' for step right
         */
        else if(key == 'd')
        {
            cam.pos[0] += step_size * cos(x_view_rad);
            cam.pos[2] += step_size * sin(x_view_rad);
            glutPostRedisplay();
        }
    }

    if (key == 'h' || key == 'u' || key == 'i' || key == 'o' || key == 'p'
        || key == 'j') {
        /* If smooth_toggle == false, revert back to the original object for each
         * object in the scene. Otherwise, smooth all the objects in the scene
         * using implicit fairing with the current time step h.
         */
        if (!smooth_toggle) {
            if (smooth_turned_off) {
                cout << "No smoothing." << endl;
                smooth_turned_off = false;
            }

            // Revert to original objects.
            objects = original_objects;
        }
        else {
            cout << "Smoothing with h = " << h << "." << endl;
            for (int i = 0; i < (int) objects.size(); i++) {
                smooth_object(objects[i], h);
            }
            cout << "Smoothing done!" << endl;
        }
        display();
    }
}

/* The 'main' function:
 *
 * Run the OpenGL program (initialize OpenGL, display the scene, and react to
 * mouse and keyboard presses).
 */
int main(int argc, char* argv[])
{
    if (argc != 5) {
        string usage_statement = (string) "Usage: " + argv[0]
                                 + " [scene_description_file.txt] [xres]"
                                 + " [yres] [h]";
        cout << usage_statement << endl;
        return 1;
    }

    xres = atoi(argv[2]);
    yres = atoi(argv[3]);
    h = atof(argv[4]);

    // mode = 0 for Gouraud shading, mode = 1 for Phong shading.
    mode = 0;

    glutInit(&argc, argv);

    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);

    glutInitWindowSize(xres, yres);

    glutInitWindowPosition(0, 0);

    glutCreateWindow("GLSL Test");

    vertProgFileName = "vertexProgram.glsl";
    fragProgFileName = "fragmentProgram.glsl";
    init(argv[1], mode);

    glutDisplayFunc(display);

    glutReshapeFunc(reshape);

    glutMouseFunc(mouse_pressed);

    glutMotionFunc(mouse_moved);

    glutKeyboardFunc(key_pressed);

    glutMainLoop();
}
