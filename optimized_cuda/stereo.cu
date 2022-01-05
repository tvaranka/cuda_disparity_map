#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

__global__
void disparity_map(unsigned char* img0, unsigned char* img1, float* means, unsigned char* disp_img0,
				   unsigned char* disp_img1, unsigned width, unsigned height){
	const unsigned B = 5;
	const unsigned filter_size = B * B;
	const unsigned d_max = 50;
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	unsigned i, j;
	float current_max0 = -1, current_max1 = -1;
    int best_d0 = 0, best_d1 = 0;
    float zncc0, zncc1, img0_mean0, img0_mean1, img1_mean0, img1_mean1,
          bot00, bot01, bot10, bot11, top0, top1, bot10_part, bot01_part;
    int wh = width * height;
    int id = y * width + x;
    int lx = 8 * threadIdx.x;
    int ly = 2 * threadIdx.y;
    const int lbs = 32 + B - 1;
    const int bs = 32 + B - 1 + d_max;

    __shared__ unsigned char s_img0[lbs * bs];
    __shared__ unsigned char s_img1[lbs * bs];

    for (int idx_x = lx; idx_x < lx + 8; idx_x++){
    	for (int idx_y = ly; idx_y < ly + 2; idx_y++){
    		int sx = (blockIdx.x * blockDim.x) + idx_x;
			int sy = (blockIdx.y * blockDim.y) + idx_y;
    		if ((sy < height) && (sx < width) && (idx_x < bs) && (idx_y < lbs)) {
	    		s_img0[idx_y * bs + idx_x] = img0[sy * width + sx];
	    		s_img1[idx_y * bs + idx_x] = img1[sy * width + sx];
	    	}
    	}
    }
    __syncthreads();

    float mean0 = 0, mean1 = 0;
    for (i = 0; i < B; i++){
        for (j = 0; j < B; j++){
            mean0 += s_img0[(threadIdx.y + i) * bs + (threadIdx.x + j)];
            mean1 += s_img1[(threadIdx.y + i) * bs + (threadIdx.x + j)];
        }
    }
    
    mean0 = mean0 / filter_size;
    mean1 = mean1 / filter_size;

    bot00 = 0, bot11 = 0;
    #pragma unroll
    for (i = 0; i < B; i++){
        for (j = 0; j < B; j++){
            bot00 += powf(s_img0[(threadIdx.y + i) * bs + (threadIdx.x + j)] - mean0, 2);
            bot11 += powf(s_img1[(threadIdx.y + i) * bs + (threadIdx.x + j)] - mean1, 2);
        }
    }

    
    means[id] = mean0;
    means[id + wh] = mean1;

    __syncthreads();
    
    if ((x < width - 1 - B - d_max) && (y < height - 1 - B) && (x >= d_max)){

    	img0_mean0 = means[id];
        img1_mean1 = means[id + wh];
	    for (int d = 0; d <= d_max; d++){
	        img0_mean1 = means[y * width + (x + d)];
            img1_mean0 = means[y * width + (x - d) + wh];
            bot01 = 0; bot10 = 0; top0 = 0; top1 = 0;
	        for (i = 0; i < B; i++){
	        	#pragma unroll
	            for (j = 0; j < B; j++){
	            	//bot10_part = s_img1[(threadIdx.y + i) * bs + (threadIdx.x + j - d)] - img1_mean0;
	            	bot10_part = img1[(y + i) * width + (x + j - d)] - img1_mean0;
	            	top0 += (s_img0[(threadIdx.y + i) * bs + (threadIdx.x + j)] - img0_mean0) * bot10_part;
                    bot10 += bot10_part * bot10_part;

                    bot01_part = s_img0[(threadIdx.y + i) * bs + (threadIdx.x + j + d)] - img0_mean1;
                    top1 += bot01_part * (s_img1[(threadIdx.y + i) * bs + (threadIdx.x + j)] - img1_mean1);
                    bot01 += bot01_part * bot01_part;
	            }
	        }
	        zncc0 = top0 / (sqrtf(bot00 * bot10));
	        zncc1 = top1 / (sqrtf(bot01 * bot11));
	        
	        if (zncc0 > current_max0){
	            current_max0 = zncc0;
	            best_d0 = d;
	        }
	            
	        if (zncc1 > current_max1){
	            current_max1 = zncc1;
	            best_d1 = d;
	        }
	    }  
	}
	disp_img0[id] = (unsigned char)(best_d0 * (255.0 / d_max));
	disp_img1[id] = (unsigned char)(best_d1 * (255.0 / d_max));
}


__global__
void threshold(unsigned char* disp_img0, unsigned char* disp_img1,
	unsigned width, unsigned height, int threshold){
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int diff;
	int id = y * width + x;

	diff = abs(disp_img0[id] - disp_img1[id]);
	disp_img0[id] = (diff < threshold) * disp_img0[id];
	//occlusion filling
	int range = 0;
	if (disp_img0[y * width + x] == 0){
		for (int i = 0; i < range; i++){
			for (int j = 0; j < range; j++){
				if ((y + i < height) && (x + j < width)){
					if (disp_img0[(y + i) * width + x + j] != 0){
						disp_img1[id] = disp_img0[(y + i) * width + x + j];
						i = j = 10000;
						break;
					}
				}
			}
		}
	}
	__syncthreads();
	disp_img0[id] = ((id > 0) && (id < width * height)) ? (disp_img1[id] + disp_img1[id + 1]
					 + disp_img1[id - 1]) / 3 : 0;
}


int main() {
	
	cv::Mat img0 = cv::imread("../../data/im0small.png", cv::IMREAD_GRAYSCALE);
	cv::Mat img1 = cv::imread("../../data/im1small.png", cv::IMREAD_GRAYSCALE);
	if (img0.empty() || img1.empty()) {
		std::cout << "Failed loading image" << std::endl;
		std::cin.get();
		return -1;
	}
	
	unsigned char *d_img0, *d_img1, *d_disp_img0, *d_disp_img1;
	float * d_means;
	const int height = img0.rows;
	const int width = img0.cols;
	cv::Mat res(height, width, CV_8UC1);
	const dim3 block(32, 32);
	const dim3 grid(width / block.x, height / block.y);
	const size_t tsize = width * height * sizeof(unsigned char);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMalloc<unsigned char>(&d_img0, tsize);
	cudaMalloc<unsigned char>(&d_img1, tsize);
	cudaMalloc<unsigned char>(&d_disp_img0, tsize);
	cudaMalloc<unsigned char>(&d_disp_img1, tsize);
	cudaMalloc<float>(&d_means, tsize * 2 * sizeof(float));
	cudaMemcpy(d_img0, img0.ptr(), tsize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_img1, img1.ptr(), tsize, cudaMemcpyHostToDevice);

	cudaEventRecord(start);
	disparity_map<<<grid, block>>>(d_img0, d_img1, d_means, d_disp_img0, d_disp_img1, width, height);
	cudaDeviceSynchronize();
    //threshold<<<grid, block>>>(d_disp_img0, d_disp_img1, width, height, 12);
    //cudaDeviceSynchronize();
	cudaEventRecord(stop);

	cudaMemcpy(res.ptr(), d_disp_img0, tsize, cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	std::cout << "Time(ms): " << milliseconds << std::endl;
	std::cout << res.size() << std::endl;
	//cv::imwrite("../depth_full_size.jpg", res);
	cv::namedWindow("Window name", cv::WINDOW_NORMAL);
	cv::imshow("Window name", res);
	cv::waitKey(0);
	cudaFree(d_img0);
	cudaFree(d_img1);
	cudaFree(d_means);
	cudaFree(d_disp_img0);
	cudaFree(d_disp_img1);
	//cv::destroyWindow(window_name);
	return 0;
}