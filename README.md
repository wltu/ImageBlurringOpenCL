# Image Blurring OpenCL
This is an OpenCL Project for Image Blurring Application comparing the performance between OpenCV builtin blur method and implementation with GPU parallel programming using OpenCL. Within the OpenCL implementation, different data structured is tested to see if can affect the memory access time and the overall performance.


### Averaging Blur
The implementation of the image blur is based on [averaging blur](https://mmeysenburg.github.io/image-processing/06-blurring/). Three different implementations were tested which includes OpenCV builtin, OpenCL with `vector<uchar>`, and OpenCL with `vector<uchar3>`.
  
### Implementation Methodology
 * Image:
 
 The image data is read and uncompressed using OpenCV's method `Mat imread(const String& filename, int flags=IMREAD_COLOR )`
 * OpenCV:
 
 
The builtin OpenCV ``` cv::blur ( InputArray src, OutputArray dst, Size ksize, Point anchor = Point(-1,-1), int borderType = BORDER_DEFAULT ) ``` was used for the base line of the tests.
 * OpenCL:
 
 
 The OpenCL implementation includes converting the including converting `Mat` data provided by the image into `vector<uchar>` and sending that to the GPU with a kernel method. Lastly, converting the `vector<uchar>` back into 'Mat'.
 * OpenCL Data Structure:
 
 OpenCL Data Structure is similar to the previous OpenCL implementation, except that the data passed into the kernel is `vector<uchar3>`. The `vector<uchar>` that was converted from `Mat` is converted into `vector<uchar3>` with another kernel method using the GPU and convert back after the blur operation is done into `Mat`.
 
### Performance Time:
The performance of each implementation is measured by the total time spent from reading the image to opening an window to show the blurred image. The operations that is included for each implementation is listed below.

 * OpenCV:
    * Read Image
    * Blur
 
 * OpenCL:
    * Read Image
    * Convert `Mat` to `vector<uchar>`
    * Set up kernel function
    * Blur (GPU)
    * Convert `vector<uchar>` to `Mat'
    
 * OpenCL Data Structure:
    * Read Image
    * Convert `Mat` to `vector<uchar>`
    * Set up kernel functions 
    * Convert `vector<uchar>` to `vector<uchar3>` (GPU)
    * Blur (GPU)
    * Convert `vector<uchar3>` to `vector<uchar>` (GPU)
    * Convert `vector<uchar>` to `Mat'
    
### Result: 

#### Original:
<p align="center">
  <img width="600" height="400" src="https://user-images.githubusercontent.com/27719918/55678620-88421a80-58b1-11e9-971a-6978a4f34098.jpg">
</p>

#### OpenCV:
<p align="center">
  <img width="600" height="400" src="https://user-images.githubusercontent.com/27719918/55678632-b0317e00-58b1-11e9-9481-c3bc36d39a98.jpg">
</p>

#### OpenCL:
<p align="center">
  <img width="600" height="400" src="https://user-images.githubusercontent.com/27719918/55678623-97c16380-58b1-11e9-9616-5206786015a3.jpg">
</p>

#### OpenCL - Data:
<p align="center">
  <img width="600" height="400" src="https://user-images.githubusercontent.com/27719918/55678628-a7d94300-58b1-11e9-8df2-ed0b91df7cf1.jpg">
</p>



##### 5x5 window

| Implementation   | Average Time      |
| ---------------- |:-----------------:| 
| OpenCV           | 0.120641 seconds  | 
| OpenCL           | 0.0454018 seconds | 
| OpenCL-Data      | 0.258974 seconds  | 

##### 9x9 window
| Implementation   | Average Time      |
| ---------------- |:-----------------:| 
| OpenCV           | 0.116062 seconds  | 
| OpenCL           | 0.044688  seconds | 
| OpenCL-Data      | 0.257319 seconds  | 

##### 15x15 window
| Implementation   | Average Time      |
| ---------------- |:-----------------:| 
| OpenCV           | 0.116528 seconds  | 
| OpenCL           | 0.116528  seconds | 
| OpenCL-Data      | 0.116528 seconds  | 

#### Average
| Implementation   | Average Time      |
| ---------------- |:-----------------:| 
| OpenCV           | 0.117744 seconds  | 
| OpenCL           | 0.045961  seconds | 
| OpenCL-Data      | 0.258592 seconds  | 

