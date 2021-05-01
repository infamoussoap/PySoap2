softmax_source_code = """
__kernel void softmax(__global const float *x, __global const float *max_val_across_prediction, 
                      __global int *inputLength, __global float *out)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int input_length = *inputLength;

    float numerator = exp(x[i*input_length + j] - max_val_across_prediction[i]);
    float denominator = 0.0;
    for (int n = 0; n < input_length; n++)
    {
        denominator += exp(x[i*input_length + n] - max_val_across_prediction[i]);
    }
    out[i*input_length + j] = numerator / denominator;
}
"""

cl_array_max_source_code = """
__kernel void max_across_last_axis(__global const float *x, __global int *inputLength, __global float *out)
{
    int i = get_global_id(0);
    int input_length = *inputLength;

    float max = x[i*input_length + 0];
    for (int n = 1; n < input_length; n++)
    {
        float x0 = x[i*input_length + n];
        max = x0 > max ? x0 : max;
    }
    out[i] = max;
}

__kernel void arg_max_across_last_axis(__global const float *x, __global int *inputLength, __global int *out)
{
    int i = get_global_id(0);
    int input_length = *inputLength;

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