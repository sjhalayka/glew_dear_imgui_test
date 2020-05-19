#version 430 core
layout(local_size_x = 1, local_size_y = 1) in;
layout(binding = 0, r32f) writeonly uniform image2D output_image;
layout(binding = 1, rgba32f) readonly uniform image2D input_image;
uniform vec4 c;
uniform int max_iterations;
uniform float threshold;

float cosh_(float x)
{
    const float e = 2.7182817459106445;
    return 0.5*(pow(e, x) + pow(e, -x));
}

float sinh_(float x)
{
    const float e = 2.7182817459106445;
    return 0.5*(pow(e, x) - pow(e, -x));
}

// A decent GLSL compiler should optimize this out.
vec4 qadd(vec4 qa, vec4 qb)
{
    return qa + qb;
}

// A decent GLSL compiler should optimize this out.
vec4 qsub(vec4 qa, vec4 qb)
{
    return qa - qb;
}

vec4 qmul(vec4 qa, vec4 qb)
{
   vec4 qout;
   qout.x = qa.x*qb.x - dot(qa.yzw, qb.yzw);
   qout.yzw = qa.x*qb.yzw + qb.x*qa.yzw + cross(qa.yzw, qb.yzw);

   return qout;
}

vec4 qdiv(vec4 qa, vec4 qb)
{
    float d = dot(qb, qb);

    vec4 temp_b = qb;
    temp_b.yzw = -temp_b.yzw;

    return qmul(temp_b / d, qa);
}

vec4 qsin(vec4 qa)
{
    float mag_vector = length(qa.yzw);

    vec4 qout;
    qout.x = sin(qa.x) * cosh_(mag_vector);
    qout.yzw = cos(qa.x) * sinh_(mag_vector) * qa.yzw / mag_vector;

    return qout;
}

vec4 qsinh(vec4 qa)
{
    float mag_vector = length(qa.yzw);

    vec4 qout;
    qout.x = sinh_(qa.x) * cos(mag_vector);
    qout.yzw = cosh_(qa.x) * sin(mag_vector) * qa.yzw / mag_vector;

    return qout;
}

vec4 qcos(vec4 qa)
{
    float mag_vector = length(qa.yzw);

    vec4 qout;
    qout.x = cos(qa.x) * cosh_(mag_vector);
    qout.yzw = -sin(qa.x) * sinh_(mag_vector) * qa.yzw / mag_vector;

    return qout;
}

vec4 qcosh(vec4 qa)
{
    float mag_vector = length(qa.yzw);

    vec4 qout;
    qout.x = cosh_(qa.x) * cos(mag_vector);
    qout.yzw = sinh_(qa.x) * sin(mag_vector) * qa.yzw / mag_vector;

    return qout;
}

vec4 qtan(vec4 qa)
{
    return qdiv(qsin(qa), qcos(qa));
}

vec4 qtanh(vec4 qa)
{
    return qdiv(qsinh(qa), qcosh(qa));
}

vec4 qpow(vec4 qa, vec4 qb)
{
    int pow_exponent = int(abs(qb.x));
    vec4 qout = qa;

    if(pow_exponent == 0)
    {
        qout.x = 1.0;
        qout.y = 0.0;
        qout.z = 0.0;
        qout.w = 0.0;
    }
    else
    {
        for(int i = 1; i < pow_exponent; i++)
            qout = qmul(qout, qa);
    }

    return qout;
}

vec4 qln(vec4 qa)
{
    qa = normalize(qa);

    float dot_vector = dot(qa.yzw, qa.yzw);
    float mag_vector = sqrt(dot_vector);

    vec4 qout;

    qout.x = 0.5 * log(qa.x*qa.x + dot_vector);
    qout.yzw = (atan(mag_vector, qa.x) * qa.yzw) / mag_vector;

    return qout;
}

vec4 qexp(vec4 qa)
{
    float mag_vector = length(qa.yzw);

    vec4 qout;
    qout.x = exp(qa.x) * cos(mag_vector);
    qout.yzw = exp(qa.x) * sin(mag_vector) * qa.yzw / mag_vector;

    return qout;
}

vec4 qsqrt(vec4 qa)
{
    vec4 qout;

    if(qa.y == 0.0 && qa.z == 0.0 && qa.w == 0.0)
    {
        if(qa.x >= 0.0)
        {
            qout.x = sqrt(qa.x);
            qout.y = 0.0;
            qout.z = 0.0;
            qout.w = 0.0;
        }
        else
        {
            qout.x = sqrt(-qa.x);
            qout.y = 0.0;
            qout.z = 0.0;
            qout.w = 0.0;
        }
    }
    else
    {
        float mag_vector = length(qa.yzw);

        if(qa.x >= 0.0)
        {
            float m = sqrt(0.5 * (sqrt(qa.x*qa.x + mag_vector*mag_vector) + qa.x));
            float l = mag_vector / (2.0 * m);
            float t = l / mag_vector;

            qout.x = m;
            qout.y = qa.y * t;
            qout.z = qa.z * t;
            qout.w = qa.w * t;
        }
        else
        {
            float l = sqrt(0.5 * (sqrt(qa.x*qa.x + mag_vector*mag_vector) - qa.x));
            float m = mag_vector / (2.0 * l);
            float t = l / mag_vector;

            qout.x = m;
            qout.y = qa.y * t;
            qout.z = qa.z * t;
            qout.w = qa.w * t;
        }
    }

    return qout;
}

vec4 qinverse(vec4 qa)
{
    vec4 qout = qa;
    qout.yzw = -qout.yzw;

    return qout / dot(qa, qa);
}

vec4 qconjugate(vec4 qa)
{
    qa.yzw = -qa.yzw;
    return qa;
}

// A decent GLSL compiler should optimize this out.
vec4 qcopy(vec4 qa)
{
    return qa;
}

vec4 qcopy_masked(vec4 qa, vec4 qb)
{
    vec4 qout;

    if(qb.x != 0.0)
    {
        if(qb.x == 1.0)
            qout.x = qa.x;
        else if(qb.x == -1.0)
            qout.x = -qa.x;
        else if(qb.x == 2.0)
            qout.x = qa.y;
        else if(qb.x == -2.0)
            qout.x = -qa.y;
        else if(qb.x == 3.0)
            qout.x = qa.z;
        else if(qb.x == -3.0)
            qout.x = -qa.z;
        else if(qb.x == 4.0)
            qout.x = qa.w;
        else if(qb.x == -4.0)
            qout.x = -qa.w;
    }

    if(qb.y != 0.0)
    {
        if(qb.y == 1.0)
            qout.y = qa.x;
        else if(qb.y == -1.0)
            qout.y = -qa.x;
        else if(qb.y == 2.0)
            qout.y = qa.y;
        else if(qb.y == -2.0)
            qout.y = -qa.y;
        else if(qb.y == 3.0)
            qout.y = qa.z;
        else if(qb.y == -3.0)
            qout.y = -qa.z;
        else if(qb.y == 4.0)
            qout.y = qa.w;
        else if(qb.y == -4.0)
            qout.y = -qa.w;
    }

    if(qb.z != 0.0)
    {
        if(qb.z == 1.0)
            qout.z = qa.x;
        else if(qb.z == -1.0)
            qout.z = -qa.x;
        else if(qb.z == 2.0)
            qout.z = qa.y;
        else if(qb.z == -2.0)
            qout.z = -qa.y;
        else if(qb.z == 3.0)
            qout.z = qa.z;
        else if(qb.z == -3.0)
            qout.z = -qa.z;
        else if(qb.z == 4.0)
            qout.z = qa.w;
        else if(qb.z == -4.0)
            qout.z = -qa.w;
    }

    if(qb.w != 0.0)
    {
        if(qb.w == 1.0)
            qout.w = qa.x;
        else if(qb.w == -1.0)
            qout.w = -qa.x;
        else if(qb.w == 2.0)
            qout.w = qa.y;
        else if(qb.w == -2.0)
            qout.w = -qa.y;
        else if(qb.w == 3.0)
            qout.w = qa.z;
        else if(qb.w == -3.0)
            qout.w = -qa.z;
        else if(qb.w == 4.0)
            qout.w = qa.w;
        else if(qb.w == -4.0)
            qout.w = -qa.w;
    }

    return qout;
}

vec4 qswizzle(vec4 qa, vec4 qb)
{
    vec4 qout;

    if(qb.x == 1.0)
        qout.x = qa.x;
    else if(qb.x == 2.0)
        qout.x = qa.y;
    else if(qb.x == 3.0)
        qout.x = qa.z;
    else
        qout.x = qa.w;

    if(qb.y == 1.0)
        qout.y = qa.x;
    else if(qb.y == 2.0)
        qout.y = qa.y;
    else if(qb.y == 3.0)
        qout.y = qa.z;
    else
        qout.y = qa.w;

    if(qb.z == 1.0)
        qout.z = qa.x;
    else if(qb.z == 2.0)
        qout.z = qa.y;
    else if(qb.z == 3.0)
        qout.z = qa.z;
    else
        qout.z = qa.w;

    if(qb.w == 1.0)
        qout.w = qa.x;
    else if(qb.w == 2.0)
        qout.w = qa.y;
    else if(qb.w == 3.0)
        qout.w = qa.z;
    else
        qout.w = qa.w;

    return qout;
}

// A decent GLSL compiler should optimize out extraneous copy calls.
vec4 iter_func(vec4 z)
{
    vec4 A0 = vec4(0, 0, 0, 0);
    vec4 A1 = vec4(0, 0, 0, 0);
    vec4 A2 = vec4(0, 0, 0, 0);
    vec4 S2_0 = vec4(0, 0, 0, 0);
    vec4 S2_1 = vec4(0, 0, 0, 0);
    vec4 S2_2 = vec4(0, 0, 0, 0);
    A0 = qcopy(z);
    A1 = qcopy(z);
    S2_0 = qsin(A0);
    S2_1 = qsin(A1);
    S2_2 = qmul(c, S2_1);
    S2_0 = qadd(S2_0, S2_2);
    A2 = qcopy(S2_0);
    z = qcopy(A2);

    return z;
}

float iterate(vec4 z)
{
    float threshold_sq = threshold*threshold;

    float len_sq = dot(z, z);

    for(int i = 0; i < max_iterations; i++)
    {
        z = iter_func(z);

        if((len_sq = dot(z, z)) >= threshold_sq)
            break;
    }

    return sqrt(len_sq);
}

void main()
{
const ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
vec4 z = imageLoad(input_image, pixel_coords);
const float magnitude = iterate(z);
const vec4 output_pixel = vec4(magnitude, 0, 0, 0);
imageStore(output_image, pixel_coords, output_pixel);
}
