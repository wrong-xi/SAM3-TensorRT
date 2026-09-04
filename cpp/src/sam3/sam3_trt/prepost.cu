#include "prepost.cuh"

__global__ void pre_process_sam3(
    uint8_t* src,
    float* dst,
    int src_width,
    int src_height,
    int src_channels,
    int dst_width,
    int dst_height)
{
    int dstX = blockIdx.x*blockDim.x + threadIdx.x;
    int dstY = blockIdx.y*blockDim.y + threadIdx.y;

    int dst_min_x = dstX*THREAD_COARSENING_FACTOR;
    int dst_min_y = dstY*THREAD_COARSENING_FACTOR;

    if (dst_min_x < dst_width && dst_min_y < dst_height)
    {
        int dst_channel_stride = dst_width*dst_height;

        #pragma unroll
        for(int ix=0; ix<THREAD_COARSENING_FACTOR; ix++)
        {
            #pragma unroll
            for (int iy=0; iy < THREAD_COARSENING_FACTOR; iy++)
            {
                int dst_loc_x = dstX*THREAD_COARSENING_FACTOR +ix;
                int dst_loc_y = dstY*THREAD_COARSENING_FACTOR +iy;

                if (dst_loc_x >= dst_width || dst_loc_y >= dst_height)
                {
                    continue;
                    // it is expected that for the vast majority of blocks 
                    // this will not cause thread divergence, except at the edges of the image
                }
                int src_loc_x = src_width*dst_loc_x/dst_width;
                src_loc_x = min(src_loc_x, src_width-1);

                int src_loc_y = src_height*dst_loc_y/dst_height;
                src_loc_y = min(src_loc_y, src_height-1);

                int dst_loc = dst_loc_y*dst_width + dst_loc_x;
                int src_loc = (src_loc_y*src_width + src_loc_x)*src_channels;

                uint8_t sb = src[src_loc];
                uint8_t sg = src[src_loc+1];
                uint8_t sr = src[src_loc+2];

                float dr = (static_cast<float>(sr)*SAM3_RESCALE_FACTOR - SAM3_IMG_MEAN)/SAM3_IMG_STD;
                float dg = (static_cast<float>(sg)*SAM3_RESCALE_FACTOR - SAM3_IMG_MEAN)/SAM3_IMG_STD;
                float db = (static_cast<float>(sb)*SAM3_RESCALE_FACTOR - SAM3_IMG_MEAN)/SAM3_IMG_STD;

                dst[dst_loc] = dr;
                dst[dst_loc + dst_channel_stride] = dg;
                dst[dst_loc + 2*dst_channel_stride] = db;
            }
        }
    }
}

// Note: In the next 2 functions, src and result matrices are assumed 
// to be the same size. It is the responsibility of the calling application
// to ensure equal sizes for these. However, mask can be a different size

__global__ void draw_fixed_prompt_semantic_masks(
    uint8_t* src,
    float* semantic_logits,
    float* presence_logits,
    uint8_t* result,
    int src_width,
    int src_height,
    int src_channels,
    int result_channels,
    int mask_width,
    int mask_height,
    float mask_alpha,
    float door_presence_threshold,
    float door_mask_threshold,
    float handle_presence_threshold,
    float handle_mask_threshold,
    float3 door_color,
    float3 handle_color)
{
    // Colors should be in 0-255 range, BGR (or the same colorspace as src).
    int resX = blockIdx.x*blockDim.x + threadIdx.x;
    int resY = blockIdx.y*blockDim.y + threadIdx.y;

    int res_min_x = resX*THREAD_COARSENING_FACTOR;
    int res_min_y = resY*THREAD_COARSENING_FACTOR;

    if (res_min_x < src_width && res_min_y < src_height)
    {
        const float door_presence_probability =
            1.0F / (1.0F + expf(-presence_logits[0]));
        const float handle_presence_probability =
            1.0F / (1.0F + expf(-presence_logits[1]));
        const int mask_area = mask_width * mask_height;

        #pragma unroll
        for (int ix=0; ix < THREAD_COARSENING_FACTOR; ix++)
        {
            #pragma unroll
            for (int iy=0; iy < THREAD_COARSENING_FACTOR; iy++)
            {
                int res_loc_x = resX*THREAD_COARSENING_FACTOR + ix;
                int res_loc_y = resY*THREAD_COARSENING_FACTOR + iy;
                if (res_loc_x >= src_width || res_loc_y >= src_height)
                {
                    continue;
                }

                const int pixel_index = res_loc_y*src_width + res_loc_x;
                const int src_loc = pixel_index*src_channels;

                int mask_loc_x = res_loc_x*mask_width/src_width;
                int mask_loc_y = res_loc_y*mask_height/src_height;
                int mask_loc = mask_loc_y*mask_width + mask_loc_x;

                const float door_mask_probability =
                    1.0F / (1.0F + expf(-semantic_logits[mask_loc]));
                const float handle_mask_probability =
                    1.0F / (1.0F + expf(
                        -semantic_logits[mask_area + mask_loc]));
                const SemanticClassSelection selection = select_semantic_class(
                    door_presence_probability,
                    door_mask_probability,
                    handle_presence_probability,
                    handle_mask_probability,
                    door_presence_threshold,
                    door_mask_threshold,
                    handle_presence_threshold,
                    handle_mask_threshold);

                if (result_channels == 1)
                {
                    result[pixel_index] = selection.class_label;
                    continue;
                }

                const int result_loc = pixel_index*result_channels;
                const float3 color =
                    selection.class_label == 1 ? handle_color : door_color;
                const float effective_alpha =
                    mask_alpha * selection.mask_probability;

                result[result_loc]= fmaxf(0,fminf(255,(1.0-effective_alpha)*static_cast<float>(src[src_loc]) + effective_alpha*color.x));
                result[result_loc+1]= fmaxf(0,fminf(255,(1.0-effective_alpha)*static_cast<float>(src[src_loc+1]) + effective_alpha*color.y));
                result[result_loc+2]= fmaxf(0,fminf(255,(1.0-effective_alpha)*static_cast<float>(src[src_loc+2]) + effective_alpha*color.z));
            }
        }
    }
}

__global__ void draw_instance_seg_mask(
    uint8_t* src,
    float* mask,
    uint8_t* result,
    int src_width,
    int src_height,
    int src_channels,
    int mask_width,
    int mask_height,
    int mask_channel_idx,
    float mask_alpha,
    float prob_threshold,
    float3* color_palette)
{
    // we use a 3D block and 2D grid
    int resX = blockIdx.x*blockDim.x + threadIdx.x;
    int resY = blockIdx.y*blockDim.y + threadIdx.y;

    float3 color = color_palette[(mask_channel_idx+threadIdx.z)%20];

    int res_min_x = resX*THREAD_COARSENING_FACTOR;
    int res_min_y = resY*THREAD_COARSENING_FACTOR;

    if (res_min_x < src_width && res_min_y < src_height)
    {
        #pragma unroll
        for (int ix=0; ix < THREAD_COARSENING_FACTOR; ix++)
        {
            #pragma unroll
            for (int iy=0; iy < THREAD_COARSENING_FACTOR; iy++)
            {
                int res_loc_x = resX*THREAD_COARSENING_FACTOR + ix;
                res_loc_x = min(res_loc_x, src_width-1);

                int res_loc_y = resY*THREAD_COARSENING_FACTOR + iy;
                res_loc_y = min(res_loc_y, src_height-1);
                
                int res_loc = (res_loc_y*src_width + res_loc_x)*src_channels;

                int mask_loc_x = res_loc_x*mask_width/src_width;
                int mask_loc_y = res_loc_y*mask_height/src_height;
                int mask_loc = mask_loc_y*mask_width + mask_loc_x + (mask_channel_idx+threadIdx.z)*mask_width*mask_height;

                float prob = 1.0/(1.0+ exp(-mask[mask_loc]));
                prob*=(prob > prob_threshold);

                float effective_alpha = mask_alpha*prob;
                
                if (effective_alpha)
                {
                    result[res_loc]= fmaxf(0,fminf(255,(1.0-effective_alpha)*static_cast<float>(src[res_loc]) + effective_alpha*color.x));
                    result[res_loc+1]= fmaxf(0,fminf(255,(1.0-effective_alpha)*static_cast<float>(src[res_loc+1]) + effective_alpha*color.y));
                    result[res_loc+2]= fmaxf(0,fminf(255,(1.0-effective_alpha)*static_cast<float>(src[res_loc+2]) + effective_alpha*color.z));
                }                

            }
        }
    }
}
