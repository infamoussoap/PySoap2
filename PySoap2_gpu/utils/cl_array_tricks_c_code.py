cl_array_max_source_code = """
__kernel void max_across_last_axis(__global const double *x, const int input_length, __global double *out)
{
    int i = get_global_id(0);

    double max = x[i*input_length + 0];
    double x0 = 0.0;
    for (int n = 1; n < input_length; n++)
    {
        x0 = x[i*input_length + n];
        max = x0 > max ? x0 : max;
    }
    out[i] = max;
}

__kernel void arg_max_across_last_axis(__global const double *x, const int input_length, __global int *out)
{
    int i = get_global_id(0);

    int arg_max = 0;
    double max_val = x[i*input_length + arg_max];
    double x0 = 0.0;
    for (int n = 1; n < input_length; n++)
    {
        x0 = x[i*input_length + n];
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
__kernel void sum_across_0_axis(__global const double *x, const int input_length, const int N, 
                                __global double *out)
{
    int i = get_global_id(0);

    double total = 0.0;
    for (int n = 0; n < N; n++)
    {
        total += x[i + n*input_length];
    }
    out[i] = total;
}
"""

mean_across_axis_c_code = """
__kernel void mean_across_0_axis(__global const double *x, const int input_length, const int N, 
                                 __global double *out)
{
    int i = get_global_id(0);

    double total = 0.0;

    for (int n = 0; n < N; n++)
    {
        total += x[i + n*input_length] / (double) N;
    }
    out[i] = total;
}
"""

var_across_axis_c_code = """
__kernel void var_across_0_axis(__global const double *x, __global const double *mean, const int input_length, 
                                const int N, __global double *out)
{
    int i = get_global_id(0);

    double total = 0.0;
    double mu = mean[i];
    
    double mean_removed = 0;
    for (int n = 0; n < N; n++)
    {
        mean_removed = x[i + n*input_length] - mu;
        total += mean_removed * mean_removed / (double) N;
    }
    out[i] = total;
}
"""

pad_images_c_code = """
__kernel void pad_images(__global const double *images, 
                         const int image_height, const int image_width, const int image_channels,
                         const int padded_image_height, const int padded_image_width,
                         const int row_start_index, const int column_start_index,
                         __global double *out)
{
    int N = get_global_id(0);

    int n = N / (image_height * image_width * image_channels);
    int i = N / (image_width * image_channels) % image_height;
    int j = N / image_channels % image_width;
    int k = N % image_channels;

    int padded_image_index = n * padded_image_height * padded_image_width * image_channels 
                             + (i + row_start_index) * padded_image_width * image_channels
                             + (j + column_start_index) * image_channels
                             + k;

    out[padded_image_index] = images[N];
}
"""
