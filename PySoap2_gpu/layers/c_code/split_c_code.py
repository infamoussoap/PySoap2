split_source_code = """
__kernel void get_input_at_mask(__global const float *input_, __global const int *mask_positions, 
                                __global int *inputLength, __global int *outputLength, __global float *output_)
{
    int input_length = *inputLength;
    int output_length = *outputLength;

    int i = get_global_id(0);
    int j = get_global_id(1);

    output_[i*output_length + j] = input_[i*input_length + mask_positions[j]];
}

__kernel void set_input_at_mask_as_output(__global float *input_, __global const int *mask_positions,
                                          __global int *inputLength, __global int *outputLength, 
                                          __global const float *output_)
{
    int input_length = *inputLength;
    int output_length = *outputLength;

    int i = get_global_id(0);
    int j = get_global_id(1);

    input_[i*input_length + mask_positions[j]] = output_[i*output_length + j];
}
"""
