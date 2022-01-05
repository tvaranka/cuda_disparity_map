
constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE
   | CLK_ADDRESS_CLAMP;

#define D_MAX 50
#define FILTER_WIDTH 5
#define WIDTH 735
#define HEIGHT 504

__kernel void disparity_map(read_only image2d_t img0,
                            read_only image2d_t img1,
                            write_only image2d_t disp_img0,
                            write_only image2d_t disp_img1) {
    //Get global id
    int x = get_global_id(0);
    int y = get_global_id(1);
    float current_max0 = -1;
    float current_max1 = -1;
    int best_d0 = 0, best_d1 = 0;
    int i, j;
    int FILTER_AREA = FILTER_WIDTH * FILTER_WIDTH;
    float zncc0, zncc1;
    //make sure we don't go out of bounds
    if ((x < WIDTH - 1 - FILTER_WIDTH - D_MAX) && (y < HEIGHT - 1 - FILTER_WIDTH)
        && (x >= D_MAX)){
        for (int d = 0; d <= D_MAX; d++){
            float img0_mean0 = 0, img0_mean1 = 0, img1_mean0 = 0, img1_mean1 = 0;

            for (i = 0; i < FILTER_WIDTH; i++){
                for (j = 0; j < FILTER_WIDTH; j++){
                    img0_mean0 += read_imagef(img0, sampler, (int2)(x + i, y + j)).s0;
                    img0_mean1 += read_imagef(img0, sampler, (int2)(x + i + d, y + j)).s0;
                    img1_mean0 += read_imagef(img1, sampler, (int2)(x + i - d, y + j)).s0;
                    img1_mean1 += read_imagef(img1, sampler, (int2)(x + i, y + j)).s0;
                }
            }

            img0_mean0 /= FILTER_AREA; img1_mean0 /= FILTER_AREA;
            img0_mean1 /= FILTER_AREA; img1_mean1 /= FILTER_AREA;
            
            float top0 = 0, top1 = 0, bot00 = 0, bot01 = 0, bot10 = 0, bot11 = 0;
            for (i = 0; i < FILTER_WIDTH; i++){
                for (j = 0; j < FILTER_WIDTH; j++){
                    top0 += (read_imagef(img0, sampler, (int2)(x + i, y + j)).s0 - img0_mean0)
                            * (read_imagef(img1, sampler, (int2)(x + i - d, y + j)).s0 - img1_mean0);
                    bot00 += pown((read_imagef(img0, sampler, (int2)(x + i, y + j)).s0 - img0_mean0), 2);
                    bot10 += pown((read_imagef(img1, sampler, (int2)(x + i - d, y + j)).s0 - img1_mean0), 2);

                    top1 += (read_imagef(img0, sampler, (int2)(x + i + d, y + j)).s0 - img0_mean1)
                            * (read_imagef(img1, sampler, (int2)(x + i, y + j)).s0 - img1_mean1);
                    bot01 += pown((read_imagef(img0, sampler, (int2)(x + i + d, y + j)).s0 - img0_mean1), 2);
                    bot11 += pown((read_imagef(img1, sampler, (int2)(x + i, y + j)).s0 - img1_mean1), 2);
                }
            }
            zncc0 = top0 / (sqrt(bot00 * bot10));
            zncc1 = top1 / (sqrt(bot01 * bot11));
            if (zncc0 > current_max0){
                current_max0 = zncc0;
                best_d0 = d;
            }
                
            if (zncc1 > current_max1){
                current_max1 = zncc1;
                best_d1 = d;
            }
        }
        write_imageui(disp_img0, (int2)(x, y), (int)(best_d0 * (255.0 / D_MAX)));
        write_imageui(disp_img1, (int2)(x, y), (int)(best_d1 * (255.0 / D_MAX)));
    }
    else{
        write_imageui(disp_img0, (int2)(x, y), 0);
        write_imageui(disp_img1, (int2)(x, y), 0);
    }
    
}
