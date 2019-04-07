#include "pch.h"

#include <iostream>
#include <math.h>
#include <string.h>
#include <vector>
#include <fstream>
#include <array>
#include<CL/cl.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <omp.h>
#include <windows.h>
#include <filesystem>

using namespace cv;
using namespace std;
using namespace std::filesystem;

// Color format: BGR

// Variables
string currentDir = "";
double OpenCVTime = 0;
double OpenCLTime = 0;
double OpenCLTimeData = 0;

vector<cl::Platform> platforms;
vector<cl::Device> devices;
cl::Platform platform;
cl::Device device;
string vender;
string version;

cl::Context context;
cl::Program program;

// Set up all OpenCL compomnets.
void setUpOpenCL();

// Get the current source directory of the project.
string getCurrentDir();

// OpenCV Builtin Blur
void blurOpenCV(Mat &image, int kernel_size, string name);

// OOpenCL implementation of image blur.
void blurOpenCL(vector<uchar> &vec, int rows, int kernel_size, string name);

// OpenCL implementation of image blur uchar3 data structure.
void blurOpenCLData(vector<uchar> &vec, int rows, int kernel_size, string name);

// Compare performace of blur image processing between OpenCV and OpenCL
void blurImageProcess(int kernel_size);


int main() {
	setUpOpenCL();
	currentDir = getCurrentDir();

	blurImageProcess(5);
	blurImageProcess(9);
	blurImageProcess(15);
}



void setUpOpenCL() {
	cl::Platform::get(&platforms);

	_ASSERT(platforms.size() > 0);
	cout << "Number of plaftorms: " << platforms.size() << endl;

	platform = platforms.front();

	platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

	_ASSERT(devices.size() > 0);
	cout << "Number of devices: " << platforms.size() << endl;

	device = devices.front();
	vender = device.getInfo<CL_DEVICE_VENDOR>();
	version = device.getInfo<CL_DEVICE_VERSION>();
	cout << vender << endl;
	cout << version << endl;

	ifstream infile("kernel.cl");
	string src(istreambuf_iterator<char>(infile), (istreambuf_iterator<char>()));
	cl::Program::Sources sources(1, make_pair(src.c_str(), src.length()));

	cl_int err = 0;

	context = cl::Context(device, 0, 0, 0, &err);

	program = cl::Program(context, sources);

	err = program.build("-cl-std=CL1.2");
}

string getCurrentDir() {
	string str;
	TCHAR path[MAX_PATH];
	GetCurrentDirectory(MAX_PATH, path);

#ifndef UNICODE
	str = path;
	return str;
#else
	std::wstring wStr = path;
	str = std::string(wStr.begin(), wStr.end());
	return str;
#endif
	return str;

}


void blurOpenCV(Mat &image, int kernel_size, string name) {
	double start = omp_get_wtime();
	blur(image, image, Size(kernel_size, kernel_size));			// OpenCV builtin image average blur.
	OpenCVTime += (omp_get_wtime() - start);

	namedWindow("Display window (OpenCV)", WINDOW_AUTOSIZE);	// Create a window for display.
	imshow("Display window (OpenCV)", image);                   // Show our image inside it.

	waitKey(0);

	// Save output image.
	if (kernel_size == 9) {
		imwrite("output/9x9_cv_" + name, image);
	}
}

void blurOpenCL(vector<uchar> &vec, int rows, int kernel_size, string name) {
	int win_size = kernel_size;

	double start = omp_get_wtime();

	kernel_size /= 2;

	int width = vec.size() / rows;
	int size = vec.size();
	vector<uchar> output(vec.size());

	cl_int err = 0;

	// Set up kernel for the GPU
	cl::Kernel kernel(program, "average_blur", &err);
	auto workGroupSize = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
	auto workGroups = vec.size() / workGroupSize;

	// Set up data buffer.
	cl::Buffer inBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(uchar) * vec.size(), vec.data(), &err);
	cl::Buffer outBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(uchar)* vec.size());

	// Set up kernel arguments.
	err = kernel.setArg(0, inBuf);
	err = kernel.setArg(1, outBuf);
	err = kernel.setArg(2, kernel_size);
	err = kernel.setArg(3, width);
	err = kernel.setArg(4, size);

	// Send Device
	cl::CommandQueue queue(context, device);
	err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(vec.size()), cl::NDRange(workGroupSize)); // Execute task
	err = queue.enqueueReadBuffer(outBuf, CL_TRUE, 0, sizeof(uchar) * output.size(), output.data()); // Copy data from global memory in GPU back into CPU memory.

	// Convert vector into Mat.
	Mat image = Mat(output).reshape(3, rows);

	OpenCLTime += (omp_get_wtime() - start);

	namedWindow("Display window (OpenCL)", WINDOW_AUTOSIZE);// Create a window for display.
	imshow("Display window (OpenCL)", image);                   // Show our image inside it.

	waitKey(0);

	if (win_size == 9) {
		imwrite("output/9x9_cl_" + name, image);
	}
}

// OpenCL with better data structure
void blurOpenCLData(vector<uchar> &vec, int rows, int kernel_size, string name) {

	int win_size = kernel_size;
	double start = omp_get_wtime();

	kernel_size /= 2;
	vector<cl_uchar3> output(vec.size() / 3);

	int width = vec.size() / rows;
	int size = vec.size();

	cl_int err = 0;
	cl::Kernel kernel(program, "convert_data", &err);
	cl::CommandQueue queue(context, device);

	auto workGroupSize = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
	auto workGroups = vec.size() / workGroupSize;

	// Convert Data to uchar3
	cl::Buffer inBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(uchar) * vec.size(), vec.data(), &err);
	cl::Buffer outBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(cl_uchar3) * output.size());

	err = kernel.setArg(0, inBuf);
	err = kernel.setArg(1, outBuf);

	err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(vec.size()), cl::NDRange(workGroupSize)); // Execute task
	err = queue.enqueueReadBuffer(outBuf, CL_TRUE, 0, sizeof(cl_uchar3) * output.size(), output.data()); // Copy data from global memory in GPU back into CPU memory.


	// Image blur.
	kernel = cl::Kernel(program, "average_blur_data", &err);

	inBuf = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(cl_uchar3) * output.size(), output.data(), &err);
	outBuf = cl::Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(cl_uchar3)* output.size());

	err = kernel.setArg(0, inBuf);
	err = kernel.setArg(1, outBuf);
	err = kernel.setArg(2, kernel_size);
	err = kernel.setArg(3, width);
	err = kernel.setArg(4, size);

	err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(output.size()), cl::NDRange(workGroupSize)); // Execute task
	err = queue.enqueueReadBuffer(outBuf, CL_TRUE, 0, sizeof(cl_uchar3) * output.size(), output.data()); // Copy data from global memory in GPU back into CPU memory.


	// Convert uchar3 data into uchar
	kernel = cl::Kernel(program, "convert_data_back", &err);

	inBuf = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(cl_uchar3) * output.size(), output.data(), &err);
	outBuf = cl::Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(uchar)* vec.size());

	err = kernel.setArg(0, inBuf);
	err = kernel.setArg(1, outBuf);

	err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(output.size()), cl::NDRange(workGroupSize)); // Execute task
	err = queue.enqueueReadBuffer(outBuf, CL_TRUE, 0, sizeof(uchar) * vec.size(), vec.data()); // Copy data from global memory in GPU back into CPU memory.
	OpenCLTimeData += omp_get_wtime() - start;


	// Convert vector into Mat.
	Mat image = Mat(vec).reshape(3, rows);

	OpenCLTimeData += omp_get_wtime() - start;

	namedWindow("Display window (OpenCL_Data)", WINDOW_AUTOSIZE);// Create a window for display.
	imshow("Display window (OpenCL_Data)", image);                   // Show our image inside it.

	waitKey(0);

	if (win_size == 9) {
		imwrite("output/9x9_cld_" + name, image);
	}
}

void blurImageProcess(int kernel_size) {
	if (kernel_size % 2 == 0) {
		cout << "Error: Kernel Size must be odd number.";

		return;
	}

	//Reset Time
	OpenCVTime = 0;
	OpenCLTime = 0;
	OpenCLTimeData = 0;


	int count = 0;
	Mat image;
	vector<uchar> imageVec;
	stringstream inputImage;
	string currentImage;
	string name;
	double start;
	double time;

	// Loop thorugh all images in the input folder.
	for (const auto & entry : directory_iterator(currentDir + "/input")) {
		inputImage << entry.path() << endl;

		currentImage = inputImage.str();

		currentImage = currentImage.substr(1, currentImage.find_last_of('"') - 1);
		name = currentImage.substr(currentImage.find_last_of('\\') + 1);

		start = omp_get_wtime();

		// Read image data into at Mat.
		image = imread(currentImage, IMREAD_COLOR);

		if (!image.data) {
			cout << "Could not open or find the image" << endl;
			break;
		}
		// Time to read the image.
		time = omp_get_wtime() - start;
		OpenCVTime += time;
		OpenCLTime += time;
		OpenCLTimeData += time;

		start = omp_get_wtime();
		imageVec.assign(image.datastart, image.dataend);	// Conver Mat into vector for the kernel to access it.
		time = omp_get_wtime() - start;						//	Time for the data structure conversion.
		OpenCLTime += time;
		OpenCLTimeData += time;

		// OpenCV
		blurOpenCV(image, kernel_size, name);


		// OpenCL
		blurOpenCL(imageVec, image.rows, kernel_size, name);

		// OpenCL with uchar3 data structure
		blurOpenCLData(imageVec, image.rows, kernel_size, name);


		inputImage.str(string());
		inputImage.clear();

		count++;
	}

	// Print out average time for the different images for a given kernel size.
	cout << "Kernel Size: " << kernel_size << " by " << kernel_size << endl;
	cout << "Time (OpenCV): " << OpenCVTime / count << " seconds" << endl;
	cout << "Time (OpenCL): " << OpenCLTime / count << " seconds" << endl;
	cout << "Time (OpenCL): " << OpenCLTimeData / count << " seconds" << endl << endl;
}