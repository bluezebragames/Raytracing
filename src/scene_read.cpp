// Philip Carr
// CS 171 Assignment 5
// November 27, 2018
// scene_read.cpp

#include "scene_read.h"

using namespace std;
using namespace Eigen;

/**
 * Return true if a given string is found in a vector of strings. Otherwise
 * return false.
 */
bool string_in_vector(const vector<string> &v, const string a) {
    for (int i = 0; i < (int) v.size(); i++) {
        if (v[i] == a) {
            return true;
        }
    }
    return false;
}

/**
 * Return the index of a given string in a vector of strings if the given string
 * is found. Otherwise, return -1 if the given string is not found in the
 * vector.
 */
int string_index_in_vector(const vector<string> &v, const string a) {
    for (int i = 0; i < (int) v.size(); i++) {
        if (v[i] == a) {
            return i;
        }
    }
    return -1;
}

/**
 * Return the vector normal to the surface defined by the given halfedge face.
 * (The returned vector is not normalized.)
 */
Vec3f calc_face_normal(HEF *face) {
    HEV *v1 = face->edge->vertex;
    HEV *v2 = face->edge->next->vertex;
    HEV *v3 = face->edge->next->next->vertex;

    Vec3f v1_to_v2;
    Vec3f v1_to_v3;

    v1_to_v2.x = v2->x - v1->x;
    v1_to_v2.y = v2->y - v1->y;
    v1_to_v2.z = v2->z - v1->z;

    v1_to_v3.x = v3->x - v1->x;
    v1_to_v3.y = v3->y - v1->y;
    v1_to_v3.z = v3->z - v1->z;

    // return cross product of v1_to_v2 and v1_to_v3
    Vec3f face_cross_product;
    face_cross_product.x = v1_to_v2.y * v1_to_v3.z - v1_to_v2.z * v1_to_v3.y;
    face_cross_product.y = v1_to_v2.z * v1_to_v3.x - v1_to_v2.x * v1_to_v3.z;
    face_cross_product.z = v1_to_v2.x * v1_to_v3.y - v1_to_v2.y * v1_to_v3.x;

    face_cross_product.x /= 2;
    face_cross_product.y /= 2;
    face_cross_product.z /= 2;

    return face_cross_product;
}

/**
 * Return the magnitude of a Vec3f struct.
 */
float vec3f_magnitude(Vec3f *v) {
    return sqrt(v->x * v->x + v->y * v->y + v->z * v->z);
}

/**
 * From halfedge.h
 *
 * Return the area-weighted normal vector of a vertex by computing a sum of the
 * face normals adjacent to the given halfedge vertex weighted by their
 * respective face areas.
 */
Vec3f calc_vertex_normal(HEV *vertex) {
      Vec3f normal;
      normal.x = 0;
      normal.y = 0;
      normal.z = 0;

      // Get outgoing halfedge from given vertex.
      HE* he = vertex->out;

      do {
          // Compute the vector normal of the plane of the face.
          Vec3f face_normal = calc_face_normal(he->face);

          // accummulate onto our normal vector
          normal.x += face_normal.x;
          normal.y += face_normal.y;
          normal.z += face_normal.z;

          // Gives us the halfedge to the next adjacent vertex.
          he = he->flip->next;
      } while(he != vertex->out);

      float normal_mag = vec3f_magnitude(&normal);
      normal.x /= normal_mag;
      normal.y /= normal_mag;
      normal.z /= normal_mag;

      return normal;
}

/**
 * Read an object's vertices and vertex normal vectors from a file.
 *
 * invalid_argument thrown if neither "v" nor "f" is found at beginning of a
 * non-empty line.
 */
Mesh_Data* read_object_file(const string filename) {
    ifstream ifs;

    // Open the file to read the object.
    ifs.open(filename.c_str());

    // Make sure the file was opened successfully.
    if (!ifs.good()) {
        throw invalid_argument("Couldn't open file");
    }

    vector<Vertex*> *vertex_vector = new vector<Vertex*>();
    vector<Face*> *face_vector = new vector<Face*>();

    /* Append a "null" vertex as the 0th element of the vertex vector so other
     * vertices can be 1-indexed.
     */
    vertex_vector->push_back(NULL);

    while (true) {
        string d_type;
        float a;
        float b;
        float c;
        ifs >> d_type;

        // Check if done reading file.
        if (ifs.eof()) {
            break;
        }

        // Add new vertex to vertex vector if "v" found at beginning of line.
        if (d_type == "v") {
            Vertex *new_vertex = new Vertex;
            ifs >> a >> b >> c;
            new_vertex->x = a;
            new_vertex->y = b;
            new_vertex->z = c;
            (*vertex_vector).push_back(new_vertex);
        }

        // Add new face to face vector if "f" found at beginning of line.
        else if (d_type == "f") {
            Face *new_face = new Face;
            ifs >> a >> b >> c;
            new_face->idx1 = (int) a;
            new_face->idx2 = (int) b;
            new_face->idx3 = (int) c;
            (*face_vector).push_back(new_face);
        }

        /* Throw error if neither "v", "vn" nor "f" is found at beginning of
         * line.
         */
        else {
            string error_message = (string) "obj file data type is neither "
                                   + "vertex nor face";
            string error_value = (string) d_type;
            throw invalid_argument(error_message + " (" + error_value + ")");
        }
    }

    // Close the file.
    ifs.close();

    Mesh_Data *mesh_data = new Mesh_Data;
    mesh_data->vertices = vertex_vector;
    mesh_data->faces = face_vector;

    return mesh_data;
}

/**
 * Set the given object's vertex and normal buffers (using the object's
 * mesh_data).
 */
void update_object_buffers(Object &obj) {
    Mesh_Data *mesh_data = obj.mesh_data;

    /* Object vertex buffer.
     */
    vector<Vertex> vertex_buffer;

    // Use halfedges to build vector of vertex normals.
    // Initialize Object normal buffer.
    vector<Vec3f> normal_buffer;

    vector<HEV*> *hevs = new vector<HEV*>();
    vector<HEF*> *hefs = new vector<HEF*>();
    build_HE(mesh_data, hevs, hefs);

    for (int i = 0; i < (int) (*(mesh_data->faces)).size(); i++) {
        int idx1 = (*(mesh_data->faces))[i]->idx1;
        int idx2 = (*(mesh_data->faces))[i]->idx2;
        int idx3 = (*(mesh_data->faces))[i]->idx3;
        Vec3f normal1 = calc_vertex_normal((*hevs)[idx1]);
        Vec3f normal2 = calc_vertex_normal((*hevs)[idx2]);
        Vec3f normal3 = calc_vertex_normal((*hevs)[idx3]);
        normal_buffer.push_back(normal1);
        normal_buffer.push_back(normal2);
        normal_buffer.push_back(normal3);
        vertex_buffer.push_back(*(*(mesh_data->vertices))[idx1]);
        vertex_buffer.push_back(*(*(mesh_data->vertices))[idx2]);
        vertex_buffer.push_back(*(*(mesh_data->vertices))[idx3]);
    }
    delete_HE(hevs, hefs);

    obj.vertex_buffer = vertex_buffer;
    obj.normal_buffer = normal_buffer;
}

/**
 * Assign each halfedge vertex in halfedge vertex vector to an index.
 */
void index_vertices(vector<HEV*> *hevs) {
    for(int i = 1; i < hevs->size(); i++)
        hevs->at(i)->index = i;
}

/**
 * Return the sum of all the face areas adjacent to a given halfedge's vertex.
 */
float calc_neighbor_area_sum(HE *he) {
    float neighbor_area_sum = 0;
    HE *start = he;
    do {
        Vec3f face_normal = calc_face_normal(he->face);
        neighbor_area_sum += vec3f_magnitude(&face_normal);
        he = he->flip->next;
    } while(he != start);
    return neighbor_area_sum;
}

/**
 * Return the cotangent of the angle alpha given halfedge vertices vi and vj.
 * The angle alpha corresponds to the angle opposite to the flipped halfedge
 * face of the given halfedge start.
 */
float calc_cot_alpha(HEV *vi, HEV *vj, HE *start) {
    HEV *v_alpha = start->flip->next->next->vertex;

    Vector3f v_alpha_to_vi;
    Vector3f v_alpha_to_vj;

    v_alpha_to_vi << vi->x - v_alpha->x, vi->y - v_alpha->y, vi->z - v_alpha->z;

    v_alpha_to_vj << vj->x - v_alpha->x, vj->y - v_alpha->y, vj->z - v_alpha->z;

    float dot_product = v_alpha_to_vi.dot(v_alpha_to_vj);
    float cross_product_mag = v_alpha_to_vi.cross(v_alpha_to_vj).norm();

    return dot_product / cross_product_mag;
}

/**
 * Return the cotangent of the angle beta given halfedge vertices vi and vj.
 * The angle beta corresponds to the angle opposite to the halfedge face of the
 * given halfedge start.
 */
float calc_cot_beta(HEV *vi, HEV *vj, HE *start) {
    HEV *v_beta = start->next->next->vertex;

    Vector3f v_beta_to_vi;
    Vector3f v_beta_to_vj;

    v_beta_to_vi << vi->x - v_beta->x, vi->y - v_beta->y, vi->z - v_beta->z;

    v_beta_to_vj << vj->x - v_beta->x, vj->y - v_beta->y, vj->z - v_beta->z;

    float dot_product = v_beta_to_vi.dot(v_beta_to_vj);
    float cross_product_mag = v_beta_to_vi.cross(v_beta_to_vj).norm();

    return dot_product / cross_product_mag;
}

/**
 * Return the F operator for implicit fairing in matrix form.
 */
SparseMatrix<float> build_F_operator(vector<HEV*> *hevs, const float h) {
    int num_vertices = hevs->size() - 1;

    // Initialize a sparse matrix to represent our F operator.
    SparseMatrix<float> F(num_vertices, num_vertices);

    // Reserve room for 7 non-zeros per row of F.
    F.reserve(VectorXi::Constant(num_vertices, 7));
    for(int i = 1; i < hevs->size(); i++) {
        HE *he = hevs->at(i)->out;

        float neighbor_area_sum = calc_neighbor_area_sum(he);

        if (neighbor_area_sum > 0.00001) {
            int neighbor_count = 0;
            float cot_alpha_sum = 0;
            float cot_beta_sum = 0;

            // Iterate over all vertices adjacent to vi.
            do {
                // Get index of adjacent vertex to vi.
                int j = he->next->vertex->index;

                // Call function to compute cotangent of angle alpha.
                float cot_alpha = calc_cot_alpha(he->vertex, he->next->vertex,
                                                 he);
                cot_alpha_sum += cot_alpha;

                // Call function to compute cotangent of angle beta.
                float cot_beta = calc_cot_beta(he->vertex, he->next->vertex,
                                               he);
                cot_beta_sum += cot_beta;

                /* Fill the j-th slot of row i of our L matrix with
                 * appropriate value.
                 */
                float i_j_term = (float) -h / 2.0 / neighbor_area_sum
                                     * (cot_alpha + cot_beta);
                F.insert(i-1, j-1) = i_j_term;

                neighbor_count++;
                he = he->flip->next;
            } while(he != hevs->at(i)->out);

            /* Fill the i-th slot of row i of our L matrix with
             * appropriate value.
             */
            float i_i_term = 1 + (float) h / 2.0 / neighbor_area_sum
                                     * (cot_alpha_sum + cot_beta_sum);
            F.insert(i-1, i-1) = i_i_term;
        }
    }

    // Make the matrix more space-efficient.
    F.makeCompressed();

    return F;
}

/**
 * Solve the nonlinear matrix equations F x_h = x_0, F y_h = y_0, and
 * F z_h = z_0, and update the halfedge vertices with the vertices
 * (x_h, y_h, z_h) corresponding to the smoothed object.
 */
void solve(vector<HEV*> *hevs, SparseMatrix<float> F) {
    // Initialize Eigen’s sparse solver.
    SparseLU<SparseMatrix<float>, COLAMDOrdering<int>> solver;

    // The following two lines essentially tailor our solver to our operator F.
    solver.analyzePattern(F);
    solver.factorize(F);

    int num_vertices = hevs->size() - 1;

    // Initialize our vector representation of x, y, or z coordinate.
    VectorXf x0_vector(num_vertices);
    VectorXf y0_vector(num_vertices);
    VectorXf z0_vector(num_vertices);

    for(int i = 1; i < hevs->size(); ++i) {
        x0_vector(i - 1) = hevs->at(i)->x;
        y0_vector(i - 1) = hevs->at(i)->y;
        z0_vector(i - 1) = hevs->at(i)->z;
    }

    // Have Eigen solve for the smoothed coordinate vectors.
    VectorXf xh_vector(num_vertices);
    VectorXf yh_vector(num_vertices);
    VectorXf zh_vector(num_vertices);

    xh_vector = solver.solve(x0_vector);
    yh_vector = solver.solve(y0_vector);
    zh_vector = solver.solve(z0_vector);

    /* Update the hevs vertices with the vertices corresponding to the smoothed
     * object.
     */
    for(int i = 1; i < hevs->size(); ++i) {
        hevs->at(i)->x = xh_vector(i - 1);
        hevs->at(i)->y = yh_vector(i - 1);
        hevs->at(i)->z = zh_vector(i - 1);
    }
}

/**
 * Smooth the given object using implicit fairing with the given time step h.
 */
void smooth_object(Object &obj, const float h) {
    Mesh_Data *mesh_data = obj.mesh_data;

    vector<HEV*> *hevs = new vector<HEV*>();
    vector<HEF*> *hefs = new vector<HEF*>();
    build_HE(mesh_data, hevs, hefs);
    index_vertices(hevs);

    // Build F operator to solve for x, y and z coordinates.
    SparseMatrix<float> F = build_F_operator(hevs, h);

    /* Solve F xh = x0, F yh = y0, F zh = z0, for xh, yh, and zh respectively,
     * and update the hevs vectors.
     */
    solve(hevs, F);

    // Update the object's buffers.

    // Object vertex buffer.
    vector<Vertex> vertex_buffer;

    // Object normal buffer.
    vector<Vec3f> normal_buffer;

    for (int i = 0; i < (int) hefs->size(); i++) {
        HEV *v1 = hefs->at(i)->edge->vertex;
        HEV *v2 = hefs->at(i)->edge->next->vertex;
        HEV *v3 = hefs->at(i)->edge->next->next->vertex;
        int idx1 = v1->index;
        int idx2 = v2->index;
        int idx3 = v3->index;

        // Use halfedge vertices to build vector of vertex normals.
        Vec3f normal1 = calc_vertex_normal(v1);
        Vec3f normal2 = calc_vertex_normal(v2);
        Vec3f normal3 = calc_vertex_normal(v3);
        normal_buffer.push_back(normal1);
        normal_buffer.push_back(normal2);
        normal_buffer.push_back(normal3);

        Vertex vertex1;
        Vertex vertex2;
        Vertex vertex3;

        vertex1.x = v1->x;
        vertex1.y = v1->y;
        vertex1.z = v1->z;

        vertex2.x = v2->x;
        vertex2.y = v2->y;
        vertex2.z = v2->z;

        vertex3.x = v3->x;
        vertex3.y = v3->y;
        vertex3.z = v3->z;

        vertex_buffer.push_back(vertex1);
        vertex_buffer.push_back(vertex2);
        vertex_buffer.push_back(vertex3);
    }
    delete_HE(hevs, hefs);

    obj.vertex_buffer = vertex_buffer;
    obj.normal_buffer = normal_buffer;
}

/**
 * Return an Object struct of an object corresponding to the object file with
 * the given filename, using halfedges to compute the vertex normals of the
 * object.
 */
Object get_object_from_file(const string filename) {
    Mesh_Data *mesh_data = read_object_file(filename);

    Object new_object;
    new_object.mesh_data = mesh_data;
    update_object_buffers(new_object);

    return new_object;
}

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
Camera read_camera(const string filename) {
    ifstream ifs;

    // Open the file to read the objects.
    ifs.open(filename.c_str());

    // Make sure the file was opened successfully.
    if (!ifs.good()) {
        throw invalid_argument("Couldn't open file");
    }

    /* Initialize phase to keep track of which part of file is being read (phase
     * of 0 indicates "camera:" token is being searched for, 1 indicates camera
     * information (position, orientation, and perspective) are being read.
     */
    int phase = 0;

    Camera cam;

    while (true) {
        if (phase == 0) {
            string token;
            ifs >> token;
            if (ifs.eof()) {
                // Throw an error if there is no "camera:" token in the file.
                string error_message = (string) "File " + filename
                                       + "contains no \"camera:\" token.";
                throw invalid_argument(error_message);
            }
            else {
                /* If the "camera:" token is found, set phase = 1 and start
                 * reading in camera information.
                 */
                if (token == "camera:") {
                    phase = 1;
                }
            }
        }
        else {
            string name;

            /* Camera information should always be 8 lines long (1 line for
             * position information, 1 line for orientation information, and 6
             * lines for perspective information).
             */
            int camera_info_lines = 8;

            float a;
            float b;
            float c;
            float d;

            for (int i = 0; i < camera_info_lines; i++) {
                ifs >> name;
                if (name == "position") {
                    ifs >> a >> b >> c;
                    cam.pos[0] = a;
                    cam.pos[1] = b;
                    cam.pos[2] = c;
                }
                else if (name == "orientation") {
                    ifs >> a >> b >> c >> d;
                    cam.ori_axis[0] = a;
                    cam.ori_axis[1] = b;
                    cam.ori_axis[2] = c;
                    cam.ori_angle = d;
                }
                else if (name == "near") {
                    ifs >> a;
                    cam.n = a;
                }
                else if (name == "far") {
                    ifs >> a;
                    cam.f = a;
                }
                else if (name == "left") {
                    ifs >> a;
                    cam.l = a;
                }
                else if (name == "right") {
                    ifs >> a;
                    cam.r = a;
                }
                else if (name == "top") {
                    ifs >> a;
                    cam.t = a;
                }
                else if (name == "bottom") {
                    ifs >> a;
                    cam.b = a;
                }
                else {
                    string error_message = (string) "unknown camera "
                                           + "information found: " + name;
                    throw invalid_argument(error_message);
                }
            }
            break;
        }
    }

    // Close the file.
    ifs.close();

    return cam;
}

/**
 * Return a vector of all the lights found in a scene description file.
 *
 * invalid_argument thrown if file cannot be opened.
 */
vector<Point_Light> read_lights(const string filename) {
    ifstream ifs;

    // Open the file to read the lights.
    ifs.open(filename.c_str());

    // Make sure the file was opened successfully.
    if (!ifs.good()) {
        throw invalid_argument("Couldn't open file");
    }

    /* Initialize phase to keep track of which part of file is being read (phase
     * of 0 indicates "light" token is being searched for, 1 indicates light
     * information (position, color, and attenuation parameter) are being read.
     */
    int phase = 0;

    vector<Point_Light> light_vector;

    while (true) {
        if (phase == 0) {
            string token;
            ifs >> token;
            if (ifs.eof()) {
                break;
            }
            else {
                /* If the "light" token is found, set phase = 1 and start
                 * reading in light information.
                 */
                if (token == "light") {
                    phase = 1;
                }
            }
        }
        else {
            Point_Light new_light;

            float a;
            float b;
            float c;
            string s1;
            float d;
            float e;
            float f;
            string s2;
            float g;

            ifs >> a >> b >> c >> s1 >> d >> e >> f >> s2 >> g;

            new_light.pos[0] = a;
            new_light.pos[1] = b;
            new_light.pos[2] = c;
            new_light.pos[3] = 1; // w coordinate
            new_light.color[0] = d;
            new_light.color[1] = e;
            new_light.color[2] = f;
            new_light.k = g;

            light_vector.push_back(new_light);

            phase = 0;
        }
    }

    // Close the file.
    ifs.close();

    return light_vector;
}

/**
 * Read all the objects (vertices, vertex normal vectors, and corresponding
 * transformations) from a file.
 *
 * invalid_argument thrown if file cannot be opened.
 *
 * invalid_argument thrown if no "objects:" token is found.
 *
 * invalid_argument thrown if neither "t", "r", "s", nor an object name is
 * found at beginning of a non-empty line.
 */
vector<Object> read_objects(const string filename) {
    ifstream ifs;

    // Open the file to read the objects.
    ifs.open(filename.c_str());

    // Make sure the file was opened successfully.
    if (!ifs.good()) {
        throw invalid_argument("Couldn't open file");
    }

    /* Initialize phase to keep track of which part of file is being read (phase
     * of 0 indicates "objects:" token is being searched for, 1 indicates object
     * files are being read, phase of 2 indicates that corresponding object
     * transformations are being read).
     */
    int phase = 0;

    /* Declare index to keep track of which object corresponds to the object
     * transformations currently being read.
     */
    int index = -1;

    vector<string> object_names;
    vector<Object> base_objects;
    vector<Object> objects;
    vector<int> name_occurances;
    int last_obj_index = (int) objects.size() - 1;

    while (true) {
        if (phase == 0) {
            string token;
            ifs >> token;
            if (ifs.eof()) {
                // Throw an error if there is no "objects:" token in the file.
                string error_message = (string) "File " + filename
                                       + "contains no \"objects:\" token.";
                throw invalid_argument(error_message);
            }
            else {
                /* If the "objects:" token is found, set phase = 1 and start
                 * reading in objects.
                 */
                if (token == "objects:") {
                    phase = 1;
                }
            }
        }
        // phase == 1 when the file has been read up to the "objects:" token.
        else if (phase == 1) {
            string name;
            ifs >> name;

            // Check if done reading file.
            if (ifs.eof()) {
                /* Return the vector of objects if no corresponding object
                 * transformations are present in the file.
                 */
                ifs.close();
                return base_objects;
            }

            /* Read object's corresponding transformations if object name found
             * for a second time.
             */
            if (string_in_vector(object_names, name)) {
                index = string_index_in_vector(object_names, name);
                objects.push_back(base_objects[index]);
                name_occurances[index]++;
                phase = 2;
            }

            /* Append object name to vector of object names and append object as
             * read from the corresponding object file (containing object's
             * vertices and faces) to vector of base objects (objects as read
             * from an object file with no corresponding transformations).
             */
            else {
                object_names.push_back(name);
                name_occurances.push_back(0);
                string filename;
                ifs >> filename;
                base_objects.push_back(get_object_from_file(filename));
            }
        }
        else {
            last_obj_index = (int) objects.size() - 1;
            string d_type;
            ifs >> d_type;

            // Check if done reading file.
            if (ifs.eof()) {
                break;
            }

            if (d_type == "ambient") {
                float r;
                float g;
                float b;
                ifs >> r >> g >> b;
                objects[last_obj_index].ambient_reflect[0] = r;
                objects[last_obj_index].ambient_reflect[1] = g;
                objects[last_obj_index].ambient_reflect[2] = b;
            }

            else if (d_type == "diffuse") {
                float r;
                float g;
                float b;
                ifs >> r >> g >> b;
                objects[last_obj_index].diffuse_reflect[0] = r;
                objects[last_obj_index].diffuse_reflect[1] = g;
                objects[last_obj_index].diffuse_reflect[2] = b;
            }

            else if (d_type == "specular") {
                float r;
                float g;
                float b;
                ifs >> r >> g >> b;
                objects[last_obj_index].specular_reflect[0] = r;
                objects[last_obj_index].specular_reflect[1] = g;
                objects[last_obj_index].specular_reflect[2] = b;
            }

            else if (d_type == "shininess") {
                float s;
                ifs >> s;
                objects[last_obj_index].shininess = s;
            }

            // Create translation matrix if "t" found.
            else if (d_type == "t") {
                float tx;
                float ty;
                float tz;
                ifs >> tx >> ty >> tz;
                Obj_Transform t_transform;
                t_transform.type = "t";
                t_transform.components[0] = tx;
                t_transform.components[1] = ty;
                t_transform.components[2] = tz;
                objects[last_obj_index].transform_sets.push_back(t_transform);
            }

            // Create rotation matrix if "r" found.
            else if (d_type == "r") {
                float rx;
                float ry;
                float rz;
                float angle;
                ifs >> rx >> ry >> rz >> angle;
                // Normalize the rotation axis
                float rotation_axis_magnitude = \
                    sqrt(rx * rx + ry * ry + rz * rz);
                rx /= rotation_axis_magnitude;
                ry /= rotation_axis_magnitude;
                rz /= rotation_axis_magnitude;
                Obj_Transform r_transform;
                r_transform.type = "r";
                r_transform.components[0] = rx;
                r_transform.components[1] = ry;
                r_transform.components[2] = rz;
                r_transform.rotation_angle = angle * 180.0 / M_PI;
                objects[last_obj_index].transform_sets.push_back(r_transform);
            }

            // Create scaling matrix if "s" found.
            else if (d_type == "s") {
                float sx;
                float sy;
                float sz;
                ifs >> sx >> sy >> sz;
                Obj_Transform s_transform;
                s_transform.type = "s";
                s_transform.components[0] = sx;
                s_transform.components[1] = sy;
                s_transform.components[2] = sz;
                objects[last_obj_index].transform_sets.push_back(s_transform);
            }

            /* Print the current object and then append the current object to
             * the vector of objects (with their corresponding transformations),
             * and then read the next object's corresponding transformations.
             */
            else if(string_in_vector(object_names, d_type)) {
                string obj_name = (string) object_names[index] + "_copy"
                                  + to_string(name_occurances[index]);
                objects[last_obj_index].name = obj_name;
                index = string_index_in_vector(object_names, d_type);
                objects.push_back(base_objects[index]);
                name_occurances[index]++;
            }

            /* Throw invalid_argument if neither "t", "r", "s", nor an object
             * name is found at beginning of a non-empty line.
             */
            else {
                string error_message = (string) "txt file data type is neither "
                                       + "vertex nor value";
                string error_value = (string) d_type;
                throw invalid_argument(error_message
                                       + " (" + error_value + ")");
            }
        }
    }

    string obj_name = (string) object_names[index] + "_copy"
                      + to_string(name_occurances[index]);
    objects[last_obj_index].name = obj_name;

    // Close the file.
    ifs.close();

    return objects;
}
