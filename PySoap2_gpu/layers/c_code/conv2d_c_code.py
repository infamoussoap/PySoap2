conv2d_source_code = """
__kernel void predict(__global const double *z, __global const double *filter_, __global const double *bias,
                     const int filter_height, const int filter_width, const int num_of_filters,
                     const int stride, const int current_filter,
                     const int image_width, const int image_depth,
                     const int output_width,
                     const int input_length, const int output_length, 
                     __global double *out)
{   
    int n = get_global_id(0);
    int m = get_global_id(1);
    int o = get_global_id(2);
    
    int image_start_position = n*input_length + o*stride*image_depth + m*stride*image_width*image_depth;
    int filter_length = filter_width*image_depth*num_of_filters;
  
    double total = 0;
    for (int i = 0; i < filter_height; i++){
        for (int j = 0; j < filter_width; j++){
            for (int k = 0; k < image_depth; k++){
                int filter_start_position = i*filter_length 
                                            + j*image_depth*num_of_filters 
                                            + k*num_of_filters 
                                            + current_filter;
                int filter_position_on_image = i*image_width*image_depth + j*image_depth + k;
                
                double image_pixel = z[image_start_position + filter_position_on_image];
                total += image_pixel * filter_[filter_start_position];
            }
        }
    }
    
    total += bias[current_filter];
    out[n*output_length + m*output_width*num_of_filters + o*num_of_filters + current_filter] = total;
}


__kernel void delta_back_prop(__global const double *delta, __global const double *eye_conv, 
                              __global const double *g_prime, const int input_length, const int output_length, 
                              __global double *out)
{   
    int n = get_global_id(0);
    
    int i = n / input_length;
    int abc = n % input_length;
  
    double total = 0;
    for (int k = 0; k < output_length; k++){
        total += delta[i * output_length + k] * eye_conv[abc * output_length + k] * g_prime[n];
    }
    
    out[n] = total;
}


__kernel void filter_gradient(__global const double *prev_z, __global const double *delta,
                              const int output_height, const int output_width, const int num_of_filters,
                              const int stride,
                              const int image_width, const int image_depth,
                              const int N, const int input_length,
                              const int filter_width,
                              __global double *out) 
{   
    int n = get_global_id(0);

    int i = n / (filter_width * image_depth * num_of_filters);
    int j = n / (image_depth * num_of_filters) % filter_width;
    int k = n / (num_of_filters) % image_depth;
    int l = n % num_of_filters;
    
    int filter_position_on_image = i * image_width * image_depth + j * image_depth + k;
    
    double total = 0;
    
    int position_on_image = 0;
    int position_on_delta = 0;
    
    for (int a = 0; a < N; a++){
        for (int b = 0; b < output_height; b++){
            for (int c = 0; c < output_width; c++){
                position_on_image = a*input_length + b*image_width*image_depth*stride + c*image_depth*stride;
                position_on_delta = num_of_filters * (a*output_height*output_width + b*output_width + c) + l;
                
                total += prev_z[position_on_image + filter_position_on_image] * delta[position_on_delta];
            }
        }
    }
    out[n] = total;
}


__kernel void bias_gradient(__global const double *delta, const int sum_length, 
                            const int num_of_filters, __global double *out) 
{   
    int j = get_global_id(0);
    
    double total = 0;
    for (int i = 0; i < sum_length; i++){
        total += delta[i * num_of_filters + j];
    }
    out[j] = total;
}
"""