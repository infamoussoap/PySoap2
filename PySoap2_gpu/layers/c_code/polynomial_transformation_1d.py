""" P1[i, :] is assumed to be the i-th polynomial """

polynomial_1d_transform_c_code = """
__kernel void polynomial_transform_1d(__global const float *P1, 
                                      __global const float *images, __global int *M1_,
                                      __global int *inputLength, __global float *out)
{
    int n = get_global_id(0);
    int i = get_global_id(1);

    int M1 = *M1_;
    int input_length = *inputLength;

    float total = 0.0;
    for (int x = 0; x < M1; x++)
    {
        total += P1[i*M1 + x] * images[n*input_length + x];
    }

    out[n*input_length + i] = total;
}

__kernel void polynomial_transform_1d_multi(__global const float *P1,
                                            __global const float *images, __global int *M1_, __global int *M2_,
                                            __global int *inputLength, __global float *out)
{
    int M1 = *M1_;
    int M2 = *M2_;
    int input_length = *inputLength;

    int n = get_global_id(0);
    int i = get_global_id(1);
    int j = get_global_id(2);

    float total = 0.0;

    for (int x = 0; x < M1; x++)
    {
        total += P1[i*M1 + x] * images[n*input_length + x*M2 + j];
    }
    out[n*input_length + i*M2 + j] = total;
}
"""
