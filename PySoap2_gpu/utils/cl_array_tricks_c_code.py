cl_array_max_source_code = """
__kernel void max_across_last_axis(__global const float *x, const int input_length, __global float *out)
{
    int i = get_global_id(0);

    float max = x[i*input_length + 0];
    for (int n = 1; n < input_length; n++)
    {
        float x0 = x[i*input_length + n];
        max = x0 > max ? x0 : max;
    }
    out[i] = max;
}

__kernel void arg_max_across_last_axis(__global const float *x, const int input_length, __global int *out)
{
    int i = get_global_id(0);

    int arg_max = 0;
    float max_val = x[i*input_length + arg_max];

    for (int n = 1; n < input_length; n++)
    {
        float x0 = x[i*input_length + n];
        if (x0 > max_val)
        {
            arg_max = n;
            max_val = x0;
        }
    }
    out[i] = arg_max;
}
"""

cl_array_sum_across_axis_source_code = """
__kernel void sum_across_0_axis(__global const float *x, const int input_length, const int N, 
                                __global float *out)
{
    int i = get_global_id(0);

    float total = 0.0;

    for (int n = 0; n < N; n++)
    {
        total += x[i + n*input_length];
    }
    out[i] = total;
}
"""

mean_across_axis_c_code = """
__kernel void mean_across_0_axis(__global const float *x, const int input_length, const int N, 
                                 __global float *out)
{
    int i = get_global_id(0);

    float total = 0.0;

    for (int n = 0; n < N; n++)
    {
        total += x[i + n*input_length] / (float) N;
    }
    out[i] = total;
}
"""

var_across_axis_c_code = """
__kernel void var_across_0_axis(__global const float *x, __global const float *mean, const int input_length, 
                                const int N, __global float *out)
{
    int i = get_global_id(0);

    float total = 0.0;
    float mu = mean[i];

    for (int n = 0; n < N; n++)
    {
        float mean_removed = x[i + n*input_length] - mu;
        total += mean_removed * mean_removed / (float) N;
    }
    out[i] = total;
}
"""
