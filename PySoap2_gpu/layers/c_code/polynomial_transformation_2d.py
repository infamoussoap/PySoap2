""" P1[i, :] is assumed to be the i-th polynomial """

polynomial_2d_transform_c_code = """
__kernel void polynomial_transform_2d(__global const float *P1, __global const float *P2, 
                                      __global const float *images, __global int *M1_, __global int *M2_, 
                                      __global int *inputLength, __global float *out)
{
    int n = get_global_id(0);
    int i = get_global_id(1);
    int j = get_global_id(2);
    int M1 = *M1_;
    int M2 = *M2_;
    int input_length = *inputLength;

    float total = 0.0;
    for (int x = 0; x < M1; x++)
    {
        for (int y = 0; y < M2; y++)
        {
            total += P1[i*M1 + x] * P2[j*M2 + y] * images[n*input_length + x*M2 + y];
        }
    }
    out[n*input_length + i*M2 + j] = total;
}

__kernel void polynomial_transform_2d_multi(__global const float *P1, __global const float *P2, 
                                            __global const float *images, __global int *M1_, __global int *M2_,
                                            __global int *M3_, __global int *inputLength, __global float *out)
{
    int M1 = *M1_;
    int M2 = *M2_;
    int M3 = *M3_;
    int input_length = *inputLength;

    int n = get_global_id(0);
    int i = get_global_id(1);

    int J = get_global_id(2);
    int j = J / M3;
    int l = J % M3;

    float total = 0.0;

    for (int x = 0; x < M1; x++)
    {
        for (int y = 0; y < M2; y++)
        {
            total += P1[i*M1 + x] * P2[j*M2 + y] * images[n*input_length + x*M2*M3 + y*M3 + l];
        }
    }
    out[n*input_length + i*M2*M3 + j*M3 + l] = total;
}
"""
