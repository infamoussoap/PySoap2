softchop_source_code = """
__kernel void delta_back_prop(__global const double *g_prime, __global const double *new_delta, 
                              __global const double *dz, __global double *out)
{
    int i = get_global_id(0);
    out[i] = new_delta[i]*g_prime[i]*dz[i];
}


__kernel void parameter_gradient(__global const double *delta, __global const double *parameter, 
                                 const int input_length, const int N, __global double *out)
{
    int i = get_global_id(0);

    double total = 0.0;
    for (int n = 0; n < N; n++)
    {
        total += delta[n*input_length + i] * parameter[n*input_length + i];
    }
    out[i] = total;
}
"""
