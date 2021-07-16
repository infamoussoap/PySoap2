softmax_source_code = """
__kernel void softmax(__global const float *x, __global const float *max_val_across_prediction, 
                      const int input_length, __global float *out)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    double numerator = exp(x[i*input_length + j] - max_val_across_prediction[i]);
    double denominator = 0.0;
    double i_th_max_val = max_val_across_prediction[i];
    
    for (int n = 0; n < input_length; n++)
    {
        denominator += exp((double) x[i*input_length + n] - i_th_max_val);
    }
    out[i*input_length + j] = numerator / denominator;
}
"""

log_softmax_source_code = """
__kernel void log_softmax(__global const float *z, __global const float *max_across_last_axis,
                          const int input_length, __global float *out)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    int n = i * input_length;

    double max_val = max_across_last_axis[i];
    double exp_sum = 0.0;

    for(int k = 0; k < input_length; k++){
        exp_sum += exp(z[n + k] - max_val);
    }
    out[n + j] = (float) ((double) z[n + j] - max_val - log(exp_sum));
}
"""
