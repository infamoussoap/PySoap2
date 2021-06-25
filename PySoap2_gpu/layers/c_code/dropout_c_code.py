dropout_source_code = """
__kernel void dropout(__global const float *z, __global const bool *mask, const int output_length, 
                      __global float *out)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    int pos = i * output_length + j;
    out[pos] = mask[j] ? z[pos] : 0;
}"""
