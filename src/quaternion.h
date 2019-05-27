// Philip Carr
// CS 171 Assignment 5
// November 27, 2018
// quaternion.h

#include <math.h>

using namespace std;

/**
 * quaternion class for storing quaternions (represented in the form
 * q = s + xi + yj + zk = s + v (dot) (i, j, k), where the vector
 * v = (x, y, z)), and their associated operations.
 */
class quaternion {
    float s;

    float v[3];

public:
    // Default constructor for a quaternion.
    quaternion();

    // 4-argument constructor for a quaternion.
    quaternion(const float s, const float v0, const float v1,
               const float v2);

    // Return the quaternion's s-value.
    float get_s() const;

    // Return the quaternion's v-vector.
    float *get_v();

    // Set the value of s of a quaternion.
    void set_s(const float s);

    // Set the vector v of a quaternion.
    void set_v(const float v0, const float v1, const float v2);

    // Add a quaternion q to the given quaternion.
    void add(quaternion &q);

    // Multiply a scalar s to the given quaternion.
    void s_mult(const float s);

    // Return the product of the quaternion q and the given quaternion.
    quaternion q_mult(quaternion &q) const;

    // Return the complex conjugate of the given quaternion.
    quaternion conj() const;

    // Return the norm of the given quaternion (sqrt(s^2 + x^2 + y^2 + z^2)).
    float norm() const;

    // Return the inverse of the given quaternion (q^*/(|q|^2)).
    quaternion inv() const;

    // Normalize the given quaternion.
    void normalize();
};
