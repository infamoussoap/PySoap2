""" P1[i, :] is assumed to be the i-th polynomial """

polynomial_2d_transform_c_code = """
__kernel void polynomial_transform_2d(__global const float *P1, __global const float *P2, 
                                      __global const float *images, const int M1, const int M2, 
                                      const int input_length, __global float *out)
{
    int n = get_global_id(0);
    int i = get_global_id(1);
    int j = get_global_id(2);

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
                                            __global const float *images, const int M1, const int M2,
                                            const int M3, const int input_length, __global float *out)
{

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
