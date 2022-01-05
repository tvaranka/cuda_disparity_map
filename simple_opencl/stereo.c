#define _CRT_SECURE_NO_WARNINGS
#define PROGRAM_FILE "../stereo.cl"
#define KERNEL_FUNC "disparity_map"

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include "../inc/lodepng.h"
#include <omp.h>

#define IMG0_FILE "../../data/im0small.png"
#define IMG1_FILE "../../data/im1small.png"
#define OUTPUT_FILE "../result.png"
#define THRESHOLD 12

#include <CL/cl.h>

/*
Lots of boilerplate code for initializations that have been copy
*/


cl_device_id create_device() {

   cl_platform_id platform;
   cl_device_id dev;
   int err;

   /* Identify a platform */
   err = clGetPlatformIDs(1, &platform, NULL);
   if(err < 0) {
      perror("Couldn't identify a platform");
      exit(1);
   } 

   /* Access a device */
   err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
   if(err == CL_DEVICE_NOT_FOUND) {
      err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
   }
   if(err < 0) {
      perror("Couldn't access any devices");
      exit(1);   
   }

   return dev;
}

cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {

   cl_program program;
   FILE *program_handle;
   char *program_buffer, *program_log;
   size_t program_size, log_size;
   int err;
   char arg[20];

   /* Read program file and place content into buffer */
   program_handle = fopen(filename, "r");
   if(program_handle == NULL) {
      perror("Couldn't find the program file");
      exit(1);
   }
   fseek(program_handle, 0, SEEK_END);
   program_size = ftell(program_handle);
   rewind(program_handle);
   program_buffer = (char*)malloc(program_size + 1);
   program_buffer[program_size] = '\0';
   fread(program_buffer, sizeof(char), program_size, program_handle);
   fclose(program_handle);

   /* Create program from file */
   program = clCreateProgramWithSource(ctx, 1, 
      (const char**)&program_buffer, &program_size, &err);
   if(err < 0) {
      perror("Couldn't create the program");
      exit(1);
   }
   free(program_buffer);


   /* Build program */
   err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
   if(err < 0) {

      /* Find size of log and print to std output */
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 
            0, NULL, &log_size);
      program_log = (char*) malloc(log_size + 1);
      program_log[log_size] = '\0';
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 
            log_size + 1, program_log, NULL);
      printf("%s\n", program_log);
      free(program_log);
      exit(1);
   }

   return program;
}

void rgb2gray(unsigned char* input, float* output, unsigned width, unsigned height){
   for (int i = 0; i < height * width; i++){
      output[i] = 0.299 * input[i * 4] + 0.587 * input[i * 4 + 1] + 0.114 * input[i * 4 + 2];

   }
}

void threshold(unsigned char* img0, unsigned char* img1, unsigned height, unsigned width, int threshold){
   /*Cross checking between the two disparity maps  img0 and img1, and occlusion
   filling of the zero values by nearest neighbor technique.

   To reuse the allocated resources the output is returned in img0. img0 is also
   copied to img1 so that it can be reused in the blurring stage.*/
   int i, j;
   int diff;
   int color_of_nn = 0;
   for (i = 0; i < height; i++){
      for (j = 0; j < width; j++){
         diff = abs(img0[i * width + j] - img1[i * width + j]);
         img0[i * width + j] = (diff < threshold) * img0[i * width + j];
         if (img0[i * width + j] == 0){
            img0[i * width + j] = color_of_nn;
         }
         else{
            color_of_nn = img0[i * width + j];
         }
         img1[i * width + j] = img0[i * width + j];
      }
   }
}
void blur(unsigned char* img0, unsigned char* img1, unsigned height, unsigned width){
   //Simples image blurring
   int i, j;
   for (i = 1; i < height - 1; i++){
      for (j = 0; j < width; j++){
         img1[i * width + j] = (img0[(i - 1) * width + j]
            + img0[i * width + j] + img0[(i + 1) * width + j]) / 3;
      }
   }
}

int main() {
   double start = omp_get_wtime();
   /* Host/device data structures */
   cl_platform_id platform;
   cl_device_id device;
   cl_context context;
   cl_command_queue queue;
   cl_int i, j, err;
   size_t global_size[2];

   /* Program/kernel data structures */
   cl_program program;
   FILE *program_handle;
   char *program_buffer, *program_log;
   size_t program_size, log_size;
   cl_kernel kernel;
   
   /* Data and buffers */
   unsigned char* rgb_img0;
   unsigned char* rgb_img1;
   float* pixels_img0;
   float* pixels_img1;
   unsigned char* pixels_disp_img0;
   unsigned char* pixels_disp_img1;
   

   cl_mem image_img0, image_img1, disp_img0, disp_img1;
   cl_image_format format;
   unsigned error;
   unsigned width, height;
   size_t origin[3], region[3];
   cl_event timing_event1;
   cl_ulong time_start1, time_end1;


   //load image
   error = lodepng_decode32_file(&rgb_img0, &width, &height, IMG0_FILE);
   error |= lodepng_decode32_file(&rgb_img1, &width, &height, IMG1_FILE);
   if(error) printf("error %u: %s\n", error, lodepng_error_text(error));

   //transform image to gray
   pixels_img0 = (float *) malloc(width * height * sizeof(float));
   pixels_img1 = (float *) malloc(width * height * sizeof(float));
   rgb2gray(rgb_img0, pixels_img0, width, height);
   rgb2gray(rgb_img1, pixels_img1, width, height);

   free(rgb_img0);
   free(rgb_img1);

   //Malloc output pixels
   pixels_disp_img0 = (unsigned char *) malloc(width * height * sizeof(unsigned char));
   pixels_disp_img1 = (unsigned char *) malloc(width * height * sizeof(unsigned char));

   device = create_device();
   context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
   if(err < 0) {
      perror("Couldn't create a context");
      exit(1);
   }

   /* Create kernel */
   program = build_program(context, device, PROGRAM_FILE);
   kernel = clCreateKernel(program, KERNEL_FUNC, &err);
   if(err < 0) {
      printf("Couldn't create a kernel: %d\n", err);
      exit(1);
   };

   /* Create input image object */
   format.image_channel_order = CL_LUMINANCE;
   format.image_channel_data_type = CL_FLOAT;
   image_img0 = clCreateImage2D(context, 
         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
         (void *)&format, width, height, 0, (void*)pixels_img0, &err);
   image_img1 = clCreateImage2D(context, 
         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
         (void *)&format, width, height, 0, (void*)pixels_img1, &err);
   if(err < 0) {
      perror("Couldn't create the image object");
      exit(1);
   };

   /* Create output image object */
   format.image_channel_data_type = CL_UNSIGNED_INT8;
   disp_img0 = clCreateImage2D(context, 
         CL_MEM_WRITE_ONLY, &format, width, 
         height, 0, NULL, &err);
   disp_img1 = clCreateImage2D(context, 
         CL_MEM_WRITE_ONLY, &format, width, 
         height, 0, NULL, &err);
   if(err < 0) {
      perror("Couldn't create the image object");
      exit(1);
   }; 

   /* Create kernel arguments */
   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &image_img0);
   err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &image_img1);
   err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &disp_img0);
   err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &disp_img1);
   if(err < 0) {
      printf("Couldn't set a kernel argument");
      exit(1);   
   }; 
   
   /* Create a CL command queue for the device*/
   queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
   if(err < 0) {
      perror("Couldn't create the command queue");
      exit(1);   
   }
   double before_kernel = omp_get_wtime();

   global_size[0] = width; global_size[1] = height;
   //global_size[0] = 2; global_size[1] = 2;
   err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, 
         NULL, 0, NULL, &timing_event1);  
   if(err < 0) {
      perror("Couldn't enqueue the kernel");
      exit(1);
   }

   /* Read the image objects */
   origin[0] = 0; origin[1] = 0; origin[2] = 0;
   region[0] = width; region[1] = height; region[2] = 1;
   err = clEnqueueReadImage(queue, disp_img0, CL_TRUE, origin, 
         region, 0, 0, (void*)pixels_disp_img0, 0, NULL, NULL);
   err |= clEnqueueReadImage(queue, disp_img1, CL_TRUE, origin, 
         region, 0, 0, (void*)pixels_disp_img1, 0, NULL, NULL);
   if(err < 0) {
      perror("Couldn't read from the image object");
      printf("%d\n", err);
      exit(1);   
   }
   double post_process_start = omp_get_wtime();
   //post processing
   threshold(pixels_disp_img0, pixels_disp_img1, height, width, THRESHOLD);
   //reuse already malloced variables by changing names for clarity
   unsigned char* disp_img = pixels_disp_img0;
   unsigned char* blurred_img = pixels_disp_img1;
   //blurring
   blur(disp_img, blurred_img, height, width);
   double post_process_end = omp_get_wtime();


   clGetEventProfilingInfo(timing_event1, CL_PROFILING_COMMAND_START,
      sizeof(time_start1), &time_start1, NULL);
   clGetEventProfilingInfo(timing_event1, CL_PROFILING_COMMAND_END,
      sizeof(time_end1), &time_end1, NULL);
   

   size_t time_res;
   clGetDeviceInfo(device, CL_DEVICE_PROFILING_TIMER_RESOLUTION,
      sizeof(time_res), &time_res, NULL);
   printf("Timing resolution: %f\n", (cl_double) time_res * (cl_double)(1e-06));

   printf("Kernel time: %f milliseconds\n", (cl_double) (time_end1 - time_start1) * (cl_double)(1e-06));

   

   error = lodepng_encode_file(OUTPUT_FILE, pixels_disp_img0, width, height, LCT_GREY, 8);
   if(error) printf("error %u: %s\n", error, lodepng_error_text(error));

   /* Deallocate resources */
   free(pixels_img0);
   free(pixels_img1);
   free(pixels_disp_img0);
   free(pixels_disp_img1);
   clReleaseMemObject(image_img0);
   clReleaseMemObject(image_img1);
   clReleaseMemObject(disp_img0);
   clReleaseMemObject(disp_img1);
   clReleaseKernel(kernel);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);

   double end = omp_get_wtime();
   printf("Before kernel: %f ms\n", (before_kernel - start) * 1000);
   printf("Post processing: %f ms\n", (post_process_end - post_process_start) * 1000);
   printf("Total time: %f ms\n", (end - start) * 1000);

   return 0;
}
