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
 
 
 The OpenCL implementation includes converting the including converting `Mat` data provided by the image into `vector<uchar>` and sending that to the GPU with a kernel method. Lastly, converting the `vector<uchar>` back into 'Mat'
 * OpenC

Result: 
