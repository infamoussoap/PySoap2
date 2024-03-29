dense_source_code = """
__kernel void predict(__global const double *z, __global const double *W, __global const double *b, 
                      const int input_length, const int output_length, __global double *out)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
  
    double total = 0;
    for (int n = 0; n < input_length; n++){
        total += W[j*input_length + n] * z[i*input_length + n];
    }
    total += b[j];
    out[i*output_length + j] = total;
}


__kernel void delta_back_prop(__global const double *g_prime, __global const double *new_delta, 
                              __global const double *W, const int input_length, 
                              const int output_length, __global double *out)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
  
    double total = 0.0;
    for (int n = 0; n < output_length; n++){
        total += new_delta[i*output_length + n] * W[n*input_length + j];
    }
    
    out[i*input_length + j] = g_prime[i*input_length + j] * total;
}


__kernel void weight_gradient(__global const double *delta, __global const double *prev_z, 
                              const int input_length, const int output_length, const int N, 
                              __global double *out)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
  
    double total = 0.0;
    for (int n = 0; n < N; n++){
        total += delta[n*output_length + i] * prev_z[n*input_length + j];
    }
    out[i*input_length + j] = total;
}


__kernel void bias_gradient(__global const double *delta, const int output_length, const int N, 
                            __global double *out)
{
    int i = get_global_id(0);
  
    double total = 0.0;
    for (int n = 0; n < N; n++){
        total += delta[n*output_length + i];
    }
    out[i] = total;
}"""