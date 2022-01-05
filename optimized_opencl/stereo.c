#define _CRT_SECURE_NO_WARNINGS
#define PROGRAM_FILE "../stereo.cl"
#define KERNEL_GRAY "rgb2gray"
#define KERNEL_DISP "disparity_map"
#define KERNEL_MEAN "calc_means"
#define KERNEL_THRESHOLD "threshold"

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include "../inc/lodepng.h"
#include <omp.h>
#include <math.h>

#define IMG0_FILE "../../data/im0small.png"
#define IMG1_FILE "../../data/im1small.png"
#define OUTPUT_FILE "../result.png"
#define THRESHOLD 12

#include <CL/cl.h>

//gcc -o stereo stereo.c lodepng.c -lOpenCL -lgomp -lm

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

int main() {
   double start = omp_get_wtime();
   /* Host/device data structures */
   cl_platform_id platform;
   cl_device_id device;
   cl_context context;
   cl_command_queue queue;
   cl_int i, j, err;
   size_t global_size[2];
   size_t local_size[2];

   /* Program/kernel data structures */
   cl_program program;
   FILE *program_handle;
   char *program_buffer, *program_log;
   size_t program_size, log_size;
   cl_kernel kernel_gray, kernel_disp, kernel_mean, kernel_threshold;
   
   /* Data and buffers */
   unsigned char* rgb_img0;
   unsigned char* rgb_img1;
   float* pixels_img0;
   float* pixels_img1;
   float* means_img;
   unsigned char* pixels_disp_img0;
   unsigned char* pixels_disp_img1;
   

   cl_mem buffer_rgb_img0, buffer_rgb_img1, buffer_img0,
      buffer_img1, buffer_disp_img0, buffer_disp_img1,
      buffer_means_img;
   cl_image_format format;
   unsigned error;
   unsigned width, height;
   size_t origin[3], region[3];
   cl_event timing_event1, timing_event2, timing_event3;
   cl_ulong time_start1, time_end1, time_start2, time_end2, time_start3, time_end3;


   //load image
   error = lodepng_decode32_file(&rgb_img0, &width, &height, IMG0_FILE);
   error |= lodepng_decode32_file(&rgb_img1, &width, &height, IMG1_FILE);
   if(error) printf("error %u: %s\n", error, lodepng_error_text(error));

   //Malloc intermidiate values
   pixels_img0 = (float *) malloc(width * height * sizeof(float));
   pixels_img1 = (float *) malloc(width * height * sizeof(float));
   means_img = (float *) malloc(width * height * sizeof(float) * 4);



   //Malloc output pixels
   pixels_disp_img0 = (unsigned char *) malloc(width * height * sizeof(unsigned char));
   pixels_disp_img1 = (unsigned char *) malloc(width * height * sizeof(unsigned char));

   device = create_device();
   context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
   if(err < 0) {
      perror("Couldn't create a context");
      exit(1);
   }
   /* Create a CL command queue for the device*/
   queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
   if(err < 0) {
      perror("Couldn't create the command queue");
      exit(1);   
   }

   /* Create kernel */
   program = build_program(context, device, PROGRAM_FILE);
   kernel_gray = clCreateKernel(program, KERNEL_GRAY, &err);
   kernel_disp = clCreateKernel(program, KERNEL_DISP, &err);
   kernel_mean = clCreateKernel(program, KERNEL_MEAN, &err);
   kernel_threshold = clCreateKernel(program, KERNEL_THRESHOLD, &err);
   if(err < 0) {
      printf("Couldn't create a kernel: %d\n", err);
      exit(1);
   };

   /* Create input image object */
   buffer_rgb_img0 = clCreateBuffer(context,
         CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(unsigned char) * width * height * 4,
         rgb_img0, &err);
   buffer_rgb_img1 = clCreateBuffer(context,
         CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(unsigned char) * width * height * 4,
         rgb_img1, &err);
   buffer_img0 = clCreateBuffer(context,
         CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, sizeof(float) * width * height,
         pixels_img0, &err);
   buffer_img1 = clCreateBuffer(context,
         CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, sizeof(float) * width * height,
         pixels_img1, &err);
   buffer_means_img = clCreateBuffer(context,
         CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, sizeof(float) * width * height * 4,
         means_img, &err);
   buffer_disp_img0 = clCreateBuffer(context,
         CL_MEM_WRITE_ONLY, sizeof(unsigned char) * width * height,
         NULL, &err);
   buffer_disp_img1 = clCreateBuffer(context,
         CL_MEM_WRITE_ONLY, sizeof(unsigned char) * width * height,
         NULL, &err);
   if(err < 0) {
      perror("Couldn't create buffer objects");
      exit(1);
   };

   /* Create kernel arguments */
   err = clSetKernelArg(kernel_gray, 0, sizeof(cl_mem), (void *) &buffer_rgb_img0);
   err |= clSetKernelArg(kernel_gray, 1, sizeof(cl_mem), (void *) &buffer_rgb_img1);
   err |= clSetKernelArg(kernel_gray, 2, sizeof(cl_mem), (void *) &buffer_img0);
   err |= clSetKernelArg(kernel_gray, 3, sizeof(cl_mem), (void *) &buffer_img1);
   if(err < 0) {
      printf("Error: %d\n", err);
      printf("Couldn't set a kernel argument");
      exit(1);   
   }; 

   global_size[0] = width; global_size[1] = height;
   //global_size[0] = 2; global_size[1] = 2;
   err = clEnqueueNDRangeKernel(queue, kernel_gray, 2, NULL, global_size, 
         NULL, 0, NULL, NULL);
   if(err < 0) {
      perror("Couldn't enqueue the kernel");
      printf("error: %d\n", err);
      exit(1);
   }

   err = clSetKernelArg(kernel_mean, 0, sizeof(cl_mem), (void *) &buffer_img0);
   err |= clSetKernelArg(kernel_mean, 1, sizeof(cl_mem), (void *) &buffer_img1);
   err |= clSetKernelArg(kernel_mean, 2, sizeof(cl_mem), (void *) &buffer_means_img);
   if(err < 0) {
      printf("Error: %d\n", err);
      printf("Couldn't set a kernel argument");
      exit(1);   
   };
   global_size[0] = width + 1; global_size[1] = height + 8;
   local_size[0] = 32; local_size[1] = 32;
   err = clEnqueueNDRangeKernel(queue, kernel_mean, 2, NULL, global_size, 
         local_size, 0, NULL, &timing_event1);
   if(err < 0) {
      perror("Couldn't enqueue the kernel");
      printf("error: %d\n", err);
      exit(1);
   }

   err = clSetKernelArg(kernel_disp, 0, sizeof(cl_mem), (void *) &buffer_img0);
   err |= clSetKernelArg(kernel_disp, 1, sizeof(cl_mem), (void *) &buffer_img1);
   err |= clSetKernelArg(kernel_disp, 2, sizeof(cl_mem), (void *) &buffer_means_img);
   err |= clSetKernelArg(kernel_disp, 3, sizeof(cl_mem), (void *) &buffer_disp_img0);
   err |= clSetKernelArg(kernel_disp, 4, sizeof(cl_mem), (void *) &buffer_disp_img1);
   if(err < 0) {
      printf("Error: %d\n", err);
      printf("Couldn't set a kernel argument");
      exit(1);   
   };

   //global_size[0] = width; global_size[1] = height;
   err = clEnqueueNDRangeKernel(queue, kernel_disp, 2, NULL, global_size, 
         NULL, 0, NULL, &timing_event2);
   if(err < 0) {
      perror("Couldn't enqueue the kernel disp");
      printf("error: %d\n", err);
      exit(1);
   }

   err = clSetKernelArg(kernel_threshold, 0, sizeof(cl_mem), (void *) &buffer_disp_img0);
   err |= clSetKernelArg(kernel_threshold, 1, sizeof(cl_mem), (void *) &buffer_disp_img1);
   if(err < 0) {
      printf("Error: %d\n", err);
      printf("Couldn't set a kernel argument");
      exit(1);   
   };

   //global_size[0] = width; global_size[1] = height;
   err = clEnqueueNDRangeKernel(queue, kernel_threshold, 2, NULL, global_size, 
         local_size, 0, NULL, &timing_event3);
   if(err < 0) {
      perror("Couldn't enqueue the kernel");
      printf("error: %d\n", err);
      exit(1);
   }

   /* Read the image objects */
   err = clEnqueueReadBuffer(queue, buffer_disp_img0, CL_TRUE, 0,
      sizeof(unsigned char) * width * height, pixels_disp_img0, 0, NULL, NULL);  

   if(err < 0) {
      perror("Couldn't read from the buffer object");
      printf("%d\n", err);
      exit(1);   
   }

   clGetEventProfilingInfo(timing_event1, CL_PROFILING_COMMAND_START,
      sizeof(time_start1), &time_start1, NULL);
   clGetEventProfilingInfo(timing_event1, CL_PROFILING_COMMAND_END,
      sizeof(time_end1), &time_end1, NULL);
   clGetEventProfilingInfo(timing_event2, CL_PROFILING_COMMAND_START,
      sizeof(time_start2), &time_start2, NULL);
   clGetEventProfilingInfo(timing_event2, CL_PROFILING_COMMAND_END,
      sizeof(time_end2), &time_end2, NULL);
   clGetEventProfilingInfo(timing_event3, CL_PROFILING_COMMAND_START,
      sizeof(time_start3), &time_start3, NULL);
   clGetEventProfilingInfo(timing_event3, CL_PROFILING_COMMAND_END,
      sizeof(time_end3), &time_end3, NULL);
   

   size_t time_res;
   clGetDeviceInfo(device, CL_DEVICE_PROFILING_TIMER_RESOLUTION,
      sizeof(time_res), &time_res, NULL);
   printf("Timing resolution: %f\n", (cl_double) time_res * (cl_double)(1e-06));

   printf("Mean kernel time: %f milliseconds\n", (cl_double)
      (time_end1 - time_start1) * (cl_double)(1e-06));
   printf("Disparity kernel time: %f milliseconds\n", (cl_double)
      (time_end2 - time_start2) * (cl_double)(1e-06));
   printf("Threshold kernel time: %f milliseconds\n", (cl_double)
      (time_end3 - time_start3) * (cl_double)(1e-06));

   

   error = lodepng_encode_file(OUTPUT_FILE, pixels_disp_img0, width, height, LCT_GREY, 8);
   if(error) printf("error %u: %s\n", error, lodepng_error_text(error));

   /* Deallocate resources */
   free(rgb_img0);
   free(rgb_img1);
   free(pixels_img0);
   free(pixels_img1);
   free(means_img);
   free(pixels_disp_img0);
   free(pixels_disp_img1);
   clReleaseMemObject(buffer_rgb_img0);
   clReleaseMemObject(buffer_rgb_img1);
   clReleaseMemObject(buffer_img0);
   clReleaseMemObject(buffer_img1);
   clReleaseMemObject(buffer_means_img);
   clReleaseMemObject(buffer_disp_img0);
   clReleaseMemObject(buffer_disp_img1);
   clReleaseKernel(kernel_gray);
   clReleaseKernel(kernel_disp);
   clReleaseKernel(kernel_mean);
   clReleaseKernel(kernel_threshold);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);

   double end = omp_get_wtime();
   printf("Total time: %f ms\n", (end - start) * 1000);

   return 0;
}
