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
