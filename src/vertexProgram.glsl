// Philip Carr
// CS 171 Assignment 5
// November 27, 2018
// vertexProgram.glsl

// Object vertex.
varying vec4 obj_v;

// Object vertex normal vector.
varying vec3 obj_n;

void main()
{
    /* Transform the vertex and normal vector by their respective
     * transformations in world space for use in the fragment shader.
     */
    obj_v = gl_ModelViewMatrix * gl_Vertex;
    obj_n = normalize(gl_NormalMatrix * gl_Normal);

    // Transform points using the model view projecttion matrix
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}
