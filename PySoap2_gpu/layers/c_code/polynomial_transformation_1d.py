""" P1[i, :] is assumed to be the i-th polynomial """

polynomial_1d_transform_c_code = """
__kernel void polynomial_transform_1d(__global const double *P1, 
                                      __global const double *images, const int M1,
                                      const int input_length, __global double *out)
{
    int n = get_global_id(0);
    int i = get_global_id(1);

    double total = 0.0;
    for (int x = 0; x < M1; x++)
    {
        total += P1[i*M1 + x] * images[n*input_length + x];
    }

    out[n*input_length + i] = total;
}

__kernel void polynomial_transform_1d_multi(__global const double *P1,
                                            __global const double *images, const int M1, const int M2,
                                            const int input_length, __global double *out)
{
    int n = get_global_id(0);
    int i = get_global_id(1);
    int j = get_global_id(2);

    double total = 0.0;

    for (int x = 0; x < M1; x++)
    {
        total += P1[i*M1 + x] * images[n*input_length + x*M2 + j];
    }
    out[n*input_length + i*M2 + j] = total;
}
"""
