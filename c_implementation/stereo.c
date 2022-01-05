#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "../inc/lodepng.h"

//define parameters for the functions
#define DISP_VALUE 50
#define WIN_SIZE 5
#define THRESHOLD 12
//define file names of images
#define IMG0 "../../data/im0small.png"
#define IMG1 "../../data/im1small.png"

//prototypes
void rgb2gray(unsigned char *imgrgb, double **img, unsigned height, unsigned width);
void disp_map(double **img0, double **img1, int **disp_img0,
	int **disp_img1, int d_max, unsigned win_size, unsigned height, unsigned width);
void zncc(double **img0, double **img1, unsigned x, unsigned y,
	int d, unsigned B, unsigned height, unsigned width, double *zncc0, double *zncc1);
void threshold(int **img0, int **img1, unsigned height, unsigned width, int threshold);
void blur(int **img0, int **img1, unsigned height, unsigned width);


void rgb2gray(unsigned char *imgrgb, double **img, unsigned height, unsigned width){
	//As input takes 1d array rgb image and transforms it to 2d array grayscale
	for (int i = 0; i < height; i++){
		for (int j = 0; j < width; j++){
			img[i][j] = 0.299 * imgrgb[i * 4 * width + j * 4] + 
			            0.587 * imgrgb[i * 4 * width + 1 + j * 4] +
			            0.114 * imgrgb[i * 4 * width + 2 + j * 4];
		}
	}
}
void zncc(double **img0, double **img1, unsigned x, unsigned y,
	int d, unsigned B, unsigned height, unsigned width, double *zncc0, double *zncc1){
	/*Calculates the zero mean normalized cross correlation between pixels (x, y) for
	images img0 and img1 with a window length of B. d corresponds to the disparity
	value between the two images. The output is given in the parameters zncc0 and
	zncc1*/
	double img0_mean0 = 0, img0_mean1 = 0, img1_mean0 = 0, img1_mean1 = 0;
	int i, j;
	for (i = 0; i < B; i++){
		for (j = 0; j < B; j++){
			img0_mean0 += img0[x + i][y + j];
			img1_mean0 += img1[x + i][y + j - d];
			img0_mean1 += img0[x + i][y + j + d];
			img1_mean1 += img1[x + i][y + j];
		}
	}
	double win_area = B * B;
	img0_mean0 /= win_area;
    img1_mean0 /= win_area;
    img0_mean1 /= win_area;
    img1_mean1 /= win_area;
    
    double top0 = 0, top1 = 0, bot00 = 0, bot01 = 0, bot10 = 0, bot11 = 0;
    for (i = 0; i < B; i++){
		for (j = 0; j < B; j++){
			top0 += (img0[x + i][y + j] - img0_mean0) * (img1[x + i][y + j - d] - img1_mean0);
			bot00 += pow(img0[x + i][y + j] - img0_mean0, 2);
            bot10 += pow(img1[x + i][y + j - d] - img1_mean0, 2);
            
            top1 += (img0[x + i][y + j + d] - img0_mean1) * (img1[x + i][y + j] - img1_mean1);
            bot01 += pow(img0[x + i][y + j + d] - img0_mean1, 2);
            bot11 += pow(img1[x + i][y + j] - img1_mean1, 2);
		}
	}
	*zncc0 = top0 / (sqrt(bot00 * bot10));
	*zncc1 = top1 / (sqrt(bot01 * bot11));
}

void disp_map(double **img0, double **img1, int **disp_img0,
	int **disp_img1, int d_max, unsigned win_size, unsigned height, unsigned width){
	/*Calculates the raw disparity map between two images img0 and img1 and places
	the disparity maps to disp_img0 where img1 is moved for the value -d.
	Similarly for disp_img1 the img0 is moved for the value +d. d_max corresponds
	to the maximum disparity between the two images to check for. win_size is the
	window size to calculate the zncc.*/
	int x, y;
	double zncc0, zncc1;
	int best_d0 = 0, best_d1 = 0;
	for (x = 0; x < height - win_size; x++){
		for (y = d_max; y < width - win_size - d_max; y++){
			double current_max0 = -1;
            double current_max1 = -1;
            for (int d = - d_max; d < d_max; d++){
            	zncc(img0, img1, x, y, d, win_size, height, width, &zncc0, &zncc1);
            	if (zncc0 > current_max0){
            		current_max0 = zncc0;
            		best_d0 = d;
            	}
            	if (zncc1 > current_max1){
            		current_max1 = zncc1;
            		best_d1 = d;
            	}
            }
            disp_img0[x][y] = abs(best_d0);
            disp_img1[x][y] = abs(best_d1);
		}
	}

}
void threshold(int **img0, int **img1, unsigned height, unsigned width, int threshold){
	/*Cross checking between the two disparity maps  img0 and img1, and occlusion
	filling of the zero values by nearest neighbor technique.

	To reuse the allocated resources the output is returned in img0. img0 is also
	copied to img1 so that it can be reused in the blurring stage.*/
	int i, j;
	int diff;
	int color_of_nn = 0;
	for (i = 0; i < height; i++){
		for (j = 0; j < width; j++){
			diff = abs(img0[i][j] - img1[i][j]);
			img0[i][j] = (diff < threshold) * img0[i][j];
			if (img0[i][j] == 0){
				img0[i][j] = color_of_nn;
			}
			else{
				color_of_nn = img0[i][j];
			}
			img1[i][j] = img0[i][j];
		}
	}
}
void blur(int **img0, int **img1, unsigned height, unsigned width){
	//Simples image blurring
	int i, j;
	for (i = 1; i < height - 1; i++){
		for (j = 0; j < width; j++){
			img1[i][j] = (img0[i - 1][j] + img0[i][j] + img0[i + 1][j]) / 3;
		}
	}
}

int main(){
	//measure time for the whole program and the kernel separately
	double start_program, start_kernel, end_program, end_kernel;
	start_program = omp_get_wtime();
	
    double cpu_time_used;
	unsigned error;
	unsigned char *img0rgb, *img1rgb;
	double **img0, **img1;
	int **disp_img0, **disp_img1;
	unsigned width, height;
	int i, j;

	//load images
	error = lodepng_decode32_file(&img0rgb, &width, &height, IMG0);
	error = lodepng_decode32_file(&img1rgb, &width, &height, IMG1);
	if(error) printf("error %u: %s\n", error, lodepng_error_text(error));

	//allocate memory for the 2d images
	img0 = (double **) malloc(height * sizeof(double *));
	img1 = (double **) malloc(height * sizeof(double *));
	disp_img0 = (int **) malloc(height * sizeof(int *));
	disp_img1 = (int **) malloc(height * sizeof(int *));
	for (i = 0; i < height; i++){
		img0[i] = (double *)malloc(width * sizeof(double));
		img1[i] = (double *)malloc(width * sizeof(double));
		disp_img0[i] = (int *)malloc(width * sizeof(int));
		disp_img1[i] = (int *)malloc(width * sizeof(int));
	}

	//make images grayscale
	rgb2gray(img0rgb, img0, height, width);
	rgb2gray(img1rgb, img1, height, width);
	//free allocated memory that is no longer required
	free(img0rgb); free(img1rgb);
	start_kernel = omp_get_wtime();
	//calculate the disparity maps between the images im0 and img1
	disp_map(img0, img1, disp_img0, disp_img1, DISP_VALUE, WIN_SIZE, height, width);
	//post processing
	threshold(disp_img0, disp_img1, height, width, THRESHOLD);
	//reuse already malloced variables by changing names for clarity
	int **disp_img = disp_img0;
	int **blurred_img = disp_img1;
	//blurring
	blur(disp_img, blurred_img, height, width);
	end_kernel = omp_get_wtime();
	
	//transfer from 2d array to 1d for saving the image
	unsigned char *img_save = (unsigned char *)malloc(width * height * sizeof(unsigned char));
	for (i = 0; i < height; i++){
		for (j = 0; j < width; j++){
			img_save[i * width + j] = (unsigned char) ((double)blurred_img[i][j] * (255 / DISP_VALUE));
		}
	}

	//save image
	error = lodepng_encode_file("../result.png", img_save, width, height, LCT_GREY, 8);
	if(error) printf("error %u: %s\n", error, lodepng_error_text(error));


	//Free allocated memory
	for (i = 0; i < height; i++){
		free(img0[i]);
		free(img1[i]);
		free(disp_img0[i]);
		free(disp_img1[i]);
	}

	free(img0); free(img1); free(disp_img0); free(disp_img1);
	end_program = omp_get_wtime();
	
	double program_time = end_program - start_program;
	double kernel_time = end_kernel - start_kernel;
	printf("Program time: %f\nKernel time:  %f\n", program_time, kernel_time);

	return 0;
}
