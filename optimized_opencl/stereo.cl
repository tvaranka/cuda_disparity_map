
#define D_MAX 50
#define FILTER_WIDTH 5
#define FILTER_AREA 1 / (FILTER_WIDTH * FILTER_WIDTH)
#define WIDTH 735
#define HEIGHT 504
#define THRESHOLD 12

__kernel void rgb2gray(__global unsigned char* img0, __global unsigned char* img1,
                       __global float* gray0, __global float* gray1){
    //Get global id
    int x = get_global_id(0);
    int y = get_global_id(1);
    int id = y * WIDTH + x;
    gray0[id] = (float)(0.299 * img0[id * 4] + 0.587 * img0[id * 4 + 1] + 0.114 * img0[id * 4 + 2]);
    gray1[id] = (float)(0.299 * img1[id * 4] + 0.587 * img1[id * 4 + 1] + 0.114 * img1[id * 4 + 2]);
}

__kernel void calc_means(__global float* gray0, __global float* gray1,
                         __global float* means){
    
    int x = get_global_id(0);
    int y = get_global_id(1);

    //Load image to shared memory
    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int local_size = get_local_size(0);
    int block_size = local_size + FILTER_WIDTH - 1;
    __local float s_img0[36 * 36];
    __local float s_img1[36 * 36];

    int row_corner = get_group_id(0) * get_local_size(0);
    int col_corner = get_group_id(1) * get_local_size(1);
    int even_idx_x = 2 * lx;
    int even_idx_y = 2 * ly;
    int odd_idx_x = even_idx_x + 1;
    int odd_idx_y = even_idx_y + 1;
    
	//load to shared memory
    for (int idx_x = even_idx_x; idx_x <= odd_idx_x; idx_x++){
        if (idx_x < 36){
            for (int idx_y = even_idx_y; idx_y <= odd_idx_y; idx_y++){
                if (idx_y < 36){
                    int sy = col_corner + idx_y;
                    int sx = row_corner + idx_x;
                    if ((sx >= 0) && (sy >= 0) && (sy < HEIGHT) && (sx < WIDTH)){
                        s_img0[idx_y * block_size + idx_x] = gray0[sy * WIDTH + sx];
                        s_img1[idx_y * block_size + idx_x] = gray1[sy * WIDTH + sx];
                    }
                    else{
                        s_img0[idx_y * block_size + idx_x] = 0;
                        s_img1[idx_y * block_size + idx_x] = 0;
                    }
                }
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //cache the mean and bot values
    float mean0 = 0, mean1 = 0;
    for (int i = 0; i < FILTER_WIDTH; i++){
        for (int j = 0; j < FILTER_WIDTH; j++){
            mean0 += s_img0[(ly + i) * block_size  + (lx + j)];
            mean1 += s_img1[(ly + i) * block_size  + (lx + j)];
        }
    }
    
    mean0 = mean0 * FILTER_AREA;
    mean1 = mean1 * FILTER_AREA;

    

    float bot0 = 0, bot1 = 0;
    //#pragma unroll
    for (int i = 0; i < FILTER_WIDTH; i++){
        for (int j = 0; j < FILTER_WIDTH; j++){
            bot0 += pown(s_img0[(ly + i) * block_size + (lx + j)] - mean0, 2);
            bot1 += pown(s_img1[(ly + i) * block_size + (lx + j)] - mean1, 2);
        }
    }
    
    means[y * WIDTH + x] = mean0;
    means[(y * WIDTH + x) + WIDTH * HEIGHT] = mean1;
    means[(y * WIDTH + x) + WIDTH * HEIGHT * 2] = bot0;
    means[(y * WIDTH + x) + WIDTH * HEIGHT * 3] = bot1;


}

__kernel void disparity_map(__global float* gray0, __global float* gray1,
                            __global float* means,
                            __global unsigned char* disp_img0, __global unsigned char* disp_img1){
    int x = get_global_id(0);
    int y = get_global_id(1);
    float current_max0 = -1;
    float current_max1 = -1;
    int best_d0 = 0, best_d1 = 0;
    int i, j;
    float zncc0, zncc1, img0_mean0, img0_mean1, img1_mean0, img1_mean1,
          bot00, bot01, bot10, bot11, top0, top1, bot10_part, bot01_part;
    int id = y * WIDTH + x;
    int wh = WIDTH * HEIGHT;

    
    if ((x < WIDTH - 1 - FILTER_WIDTH - D_MAX) && (y < HEIGHT - 1 - FILTER_WIDTH)
        && (x >= D_MAX)){

        img0_mean0 = means[id];
        img1_mean1 = means[id + wh];
        bot00 = means[id + wh * 2];
        bot11 = means[id + wh * 3];
        
        for (int d = 0; d <= D_MAX; d++){
        	//use the cached values
            img0_mean1 = means[y * WIDTH + (x + d)];
            img1_mean0 = means[y * WIDTH + (x - d) + wh];
            bot01 = 0; bot10 = 0; top0 = 0; top1 = 0;
            #pragma unroll
            for (i = 0; i < FILTER_WIDTH; i++){
                for (j = 0; j < FILTER_WIDTH; j++){
                    bot10_part = gray1[(y + i) * WIDTH + (x + j - d)] - img1_mean0;
                    top0 += (gray0[(y + i) * WIDTH + (x + j)] - img0_mean0) * bot10_part;
                    bot10 += pown(bot10_part, 2);

                    bot01_part = gray0[(y + i) * WIDTH + (x + j + d)] - img0_mean1;
                    top1 += bot01_part * (gray1[(y + i) * WIDTH + (x + j)] - img1_mean1);
                    bot01 += pown(bot01_part, 2);
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

        disp_img0[id] = (int)(best_d0 * (255.0 / D_MAX));
        disp_img1[id] = (int)(best_d1 * (255.0 / D_MAX));

    }
    else{
        disp_img0[id] = 0;
        disp_img1[id] = 0;
    }
}

__kernel void threshold(__global unsigned char* disp_img0, __global unsigned char* disp_img1){
    int x = get_global_id(0);
    int y = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);
    int id = y * WIDTH + x;
    int diff;
    int color_of_nn = 0;

    //Load image to shared memory
    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int local_size = get_local_size(0);
    int block_size = local_size + 20;
    int lid = ly * block_size + lx;
    __local unsigned char s_img0[52 * 52];
    __local unsigned char s_img1[52 * 52];

    int row_corner = get_group_id(0) * get_local_size(0);
    int col_corner = get_group_id(1) * get_local_size(1);
    int even_idx_x = 2 * lx;
    int even_idx_y = 2 * ly;
    int odd_idx_x = even_idx_x + 1;
    int odd_idx_y = even_idx_y + 1;

    #pragma unroll
    for (int idx_x = even_idx_x; idx_x <= odd_idx_x; idx_x++){
        if (idx_x < 52){
            for (int idx_y = even_idx_y; idx_y <= odd_idx_y; idx_y++){
                if (idx_y < 52){
                    int sy = col_corner + idx_y;
                    int sx = row_corner + idx_x;
                    if ((sx >= 0) && (sy >= 0) && (sy < HEIGHT) && (sx < WIDTH)){
                        s_img0[idx_y * block_size + idx_x] = disp_img0[sy * WIDTH + sx];
                        s_img1[idx_y * block_size + idx_x] = disp_img1[sy * WIDTH + sx];
                    }
                    else{
                        s_img0[idx_y * block_size + idx_x] = 0;
                        s_img1[idx_y * block_size + idx_x] = 0;
                    }
                }
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    diff = abs(s_img0[lid] - s_img1[lid]);
    disp_img0[id] = (diff < THRESHOLD) * s_img0[lid];

    int range = 20;
    #pragma unroll
    if (s_img0[lid] == 0){
        for (int i = 0; i < range; i++){
            for (int j = 0; j < range; j++){
                if ((y + i < HEIGHT) && (x + j < WIDTH)){
                    if (s_img0[(ly + i) * block_size + lx + j] != 0){
                        s_img1[lid] = s_img0[(ly + i) * block_size + lx + j];
                        i = j = 1000;
                        break;
                    }
                }
            }
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    disp_img0[id] = (s_img1[lid] + s_img1[(ly) * block_size + lx + 1]
                    + s_img1[(ly) * block_size + lx + 2]) / 3;
}
