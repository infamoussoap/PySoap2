maxpool_source_code = """
__kernel void add_at(__global double *z, __global const int *index, __global const double *vals)
{
    int i = get_global_id(0);
    int j = index[i];
    
    z[j] += vals[i];
}


__kernel void maxpool_2d(__global const double *z,
                         const int image_height, const int image_width, const int num_channels,
                         const int window_height, const int window_width, const int stride, 
                         const int current_channel,
                         const int out_height, const int out_width,
                         const int input_length, const int out_length, 
                         __global double *max_out, __global int *argmax_out)
{   
    int n = get_global_id(0);
    int m = get_global_id(1);
    int o = get_global_id(2);

    int image_start_position = n * input_length 
                               + m * stride * image_width * num_channels 
                               + o * stride * num_channels;

    int initialized_max_val = 0;
    double max_val = 0;
    int max_arg = 0;

    int position_on_image = 0;
    int current_index = 0;
    double current_value = 0.0;
    for (int i = 0; i < window_height; i++){
        for (int j = 0; j < window_width; j++){
            position_on_image = i * image_width * num_channels + j * num_channels + current_channel;

            current_index = image_start_position + position_on_image;
            current_value = z[current_index];
            if (initialized_max_val == 0){
                max_val = current_value;
                max_arg = current_index;
                initialized_max_val = 1;
            } else {
                max_val = max_val > current_value ? max_val : current_value;
                max_arg = max_val > current_value ? max_arg : current_index;
            }
        }
    }

    int out_index = n*out_length + m*out_width*num_channels + o*num_channels + current_channel;
    max_out[out_index] = max_val;
    argmax_out[out_index] = max_arg;
}
"""