multi_softchop_source_code = """
#define SIGMOID(z) (z > 0.0 ? 1.0/(1.0 + exp(-z)) : exp(z) / (exp(z) + 1.0))

__kernel void sigmoid(__global const double *x, __global double *out)
{
    int i = get_global_id(0);
    out[i] = SIGMOID(x[i]);
}

__kernel void softchop_eval(__global const double *x, __global const double *a1, __global const double *a2,
                            __global const double *epsilon1, __global const double *epsilon2, 
                            __global int *inputLength, __global double *out)
{
    int input_length = *inputLength;
    int i = get_global_id(0);
    int j = get_global_id(1);

    int index = i*input_length + j;

    double x0 = x[index];
    out[index] = x0 * (1.0 - (1.0 - SIGMOID((x0 - a1[j])/epsilon1[j])) * SIGMOID((x0 + a2[j])/epsilon2[j]));
}

__kernel void softchop_dx(__global const double *x, __global const double *a1, __global const double *a2,
                          __global const double *epsilon1, __global const double *epsilon2, 
                          __global int *inputLength, __global double *out)
{
    int input_length = *inputLength;
    int i = get_global_id(0);
    int j = get_global_id(1);

    int index = i*input_length + j;

    double x0 = x[index];

    double sig1 = SIGMOID((x0 - a1[j])/epsilon1[j]);
    double sig2 = SIGMOID((x0 + a2[j])/epsilon2[j]);

    double sig1_minus = SIGMOID(-(x0 - a1[j])/epsilon1[j]);
    double sig2_minus = SIGMOID(-(x0 + a2[j])/epsilon2[j]);

    double summand1 = 1 - (1 - sig1) * sig2;
    double summand2 = -sig1_minus * sig1 * sig2 / epsilon1[j];
    double summand3 = (1 - sig1) * sig2_minus * sig2 / epsilon2[j];

    out[index] = summand1 - x0 * (summand2 + summand3);
}

__kernel void softchop_da1(__global const double *x, __global const double *a1, __global const double *a2,
                           __global const double *epsilon1, __global const double *epsilon2, 
                           __global int *inputLength, __global double *out)
{
    int input_length = *inputLength;
    int i = get_global_id(0);
    int j = get_global_id(1);

    int index = i*input_length + j;

    double x0 = x[index];

    double sig1 = SIGMOID((x0 - a1[j])/epsilon1[j]);
    double sig2 = SIGMOID((x0 + a2[j])/epsilon2[j]);

    double sig1_minus = SIGMOID(-(x0 - a1[j])/epsilon1[j]);

    out[index] =  -x0 * sig1_minus * sig1 * sig2 / epsilon1[j];
}

__kernel void softchop_da2(__global const double *x, __global const double *a1, __global const double *a2,
                           __global const double *epsilon1, __global const double *epsilon2, 
                           __global int *inputLength, __global double *out)
{
    int input_length = *inputLength;
    int i = get_global_id(0);
    int j = get_global_id(1);

    int index = i*input_length + j;

    double x0 = x[index];

    double sig1 = SIGMOID((x0 - a1[j])/epsilon1[j]);
    double sig2 = SIGMOID((x0 + a2[j])/epsilon2[j]);

    double sig2_minus = SIGMOID(-(x0 + a2[j])/epsilon2[j]);

    out[index] =  -x0 * (1 - sig1) * sig2 * sig2_minus / epsilon2[j];
}

__kernel void softchop_de1(__global const double *x, __global const double *a1, __global const double *a2,
                           __global const double *epsilon1, __global const double *epsilon2, 
                           __global int *inputLength, __global double *out)
{
    int input_length = *inputLength;
    int i = get_global_id(0);
    int j = get_global_id(1);

    int index = i*input_length + j;

    double x0 = x[index];

    double sig1 = SIGMOID((x0 - a1[j])/epsilon1[j]);
    double sig2 = SIGMOID((x0 + a2[j])/epsilon2[j]);

    double sig1_minus = SIGMOID(-(x0 - a1[j])/epsilon1[j]);

    out[index] =  -x0 * (x0 - a1[j]) * sig1_minus * sig1 * sig2 / (epsilon1[j] * epsilon1[j]);
}

__kernel void softchop_de2(__global const double *x, __global const double *a1, __global const double *a2,
                           __global const double *epsilon1, __global const double *epsilon2, 
                           __global int *inputLength, __global double *out)
{
    int input_length = *inputLength;
    int i = get_global_id(0);
    int j = get_global_id(1);

    int index = i*input_length + j;

    double x0 = x[index];

    double sig1 = SIGMOID((x0 - a1[j])/epsilon1[j]);
    double sig2 = SIGMOID((x0 + a2[j])/epsilon2[j]);

    double sig2_minus = SIGMOID(-(x0 + a2[j])/epsilon2[j]);

    out[index] =  x0 * (x0 + a2[j]) * (1 - sig1) * sig2 * sig2_minus / (epsilon2[j] * epsilon2[j]);
}
"""
