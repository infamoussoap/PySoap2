softchop_source_code = """
__kernel void delta_back_prop(__global const float *g_prime, __global const float *new_delta, 
                              __global const float *dz, __global float *out)
{
    int i = get_global_id(0);
    out[i] = new_delta[i]*g_prime[i]*dz[i];
}


__kernel void parameter_gradient(__global cost float *delta, __global const float *parameter, 
                                 __global int *inputLength, __global int *N1, __global float *out)
{
    int input_length = *inputLength;
    int N = *N1
    int i = get_global_id(0);

    float total = 0.0;
    for (int n = 0; n < N1; n++)
    {
        total += delta[n*input_length + i] * parameter[n*input_length + i];
    }
    out[i] = total;
}
"""
