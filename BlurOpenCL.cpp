#include "pch.h"
#include <iostream>
#include <math.h>
#include <string.h>
#include <vector>
#include <fstream>
#include <array>
#include<CL/cl.hpp>

//#include "opencv2/imgproc.hpp"
//#include "opencv2/highgui.hpp"

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


string currentDir = "";
double OpenCVTime = 0;
double OpenCLTime = 0;
int rows = -1;

vector<cl::Platform> platforms;
vector<cl::Device> devices;
cl::Platform platform;
cl::Device device;
string vender;
string version;

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

	cout << "Hello World!" << endl;
}

void OpenCLTest() {
	//vector<cl::Platform> platforms;
	//vector<cl::Device> devices;
	cl::Platform::get(&platforms);

	_ASSERT(platforms.size() > 0);
	cout << "Number of plaftorms: " << platforms.size() << endl;


	auto platform = platforms.front();

	platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

	_ASSERT(devices.size() > 0);
	cout << "Number of devices: " << platforms.size() << endl;

	auto device = devices.front();
	auto vender = device.getInfo<CL_DEVICE_VENDOR>();
	auto version = device.getInfo<CL_DEVICE_VERSION>();
	cout << vender << endl;
	cout << version << endl;

	cout << "Hello World!" << endl;


	ifstream infile("kernel.cl");
	string src(istreambuf_iterator<char>(infile), (istreambuf_iterator<char>()));
	cl::Program::Sources sources(1, make_pair(src.c_str(), src.length()));

	cl_int err = 0;

	cl::Context context(device, 0, 0, 0, &err);

	cl::Program program(context, sources);

	err = program.build("-cl-std=CL1.2");

	vector<int> vec(1024);

	for (int i = 0; i < 1024; i++) {
		vec[i] = i;
	}

	// int arr[3][2]
	const int numRow = 2;
	const int numCol = 3;
	const int size = numCol * numRow;
	array<array<int, numRow>, numCol> arr = { { {1,1}, {2,2}, {3,3}} };

	err = 0;
	cl::Kernel kernel(program, "NumericalReduction", &err);
	auto workGroupSize = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
	auto workGroups = vec.size() / workGroupSize;


	cl::Buffer inBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(int) * vec.size(), vec.data(), &err);
	cl::Buffer outBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(int)* workGroups);


	err = kernel.setArg(0, inBuf);
	err = kernel.setArg(1, workGroupSize * sizeof(int), nullptr);
	err = kernel.setArg(2, outBuf);

	vector<int> output(workGroups);

	// Send Device
	cl::CommandQueue queue(context, device);
	err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(vec.size()), cl::NDRange(workGroupSize)); // Execute task
	err = queue.enqueueReadBuffer(outBuf, CL_TRUE, 0, sizeof(int) * output.size(), output.data()); // Copy data from global memory in GPU back into CPU memory.

	for (int& out : output) {
		cout << out << endl;
	}
}

void OpenCVTest() {
	Mat image, background;
	vector<uchar> imageVec;
	vector<uchar> backVec;
	


	image = imread("input/greenscreen/greenscreen.jpg", IMREAD_COLOR);   // Read the file
	background = imread("input/background/background.jpg", IMREAD_COLOR);

	if (!image.data || !background.data){
		cout << "Could not open or find the image" << endl;
		return;
	}
	

	imageVec.assign(image.datastart, image.dataend);
	backVec.assign(background.datastart, background.dataend);

	rows = image.rows;

	double start = omp_get_wtime();

	if (backVec.size() == imageVec.size()) {
		for (int i = 0; i < imageVec.size(); i += 3) {
			if (imageVec[i] < 50 && imageVec[i + 2] < 50 && imageVec[i + 1] > 200) {
				imageVec[i] = backVec[i];
				imageVec[i + 1] = backVec[i + 1];
				imageVec[i + 2] = backVec[i + 2];
			}
		}
	}
	
	cout << "Time: " << omp_get_wtime() - start << " seconds" << endl;

	image = Mat(imageVec).reshape(3, rows); 

	namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
	imshow("Display window", image);                   // Show our image inside it.

	waitKey(0);

	//imwrite("greenscreen_output.jpg", image);
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

// OpenCV Blur Application
void blurOpenCV(Mat &image) {
	double start = omp_get_wtime();
	blur(image, image, Size(10, 10));
	OpenCVTime += (omp_get_wtime() - start);

	namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
	imshow("Display window", image);                   // Show our image inside it.

	waitKey(0);
}

// Test OpenCL application on image.
void PassFilter(vector<uchar>& vec, int rows) {
	setUpOpenCL();

	ifstream infile("kernel.cl");
	string src(istreambuf_iterator<char>(infile), (istreambuf_iterator<char>()));
	cl::Program::Sources sources(1, make_pair(src.c_str(), src.length()));

	cl_int err = 0;

	cl::Context context(device, 0, 0, 0, &err);

	cl::Program program(context, sources);

	err = program.build("-cl-std=CL1.2");

	err = 0;
	cl::Kernel kernel(program, "pass_filter", &err);
	auto workGroupSize = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
	auto workGroups = vec.size() / workGroupSize;


	cl::Buffer inBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(uchar) * vec.size(), vec.data(), &err);
	cl::Buffer outBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(uchar)* vec.size());


	err = kernel.setArg(0, inBuf);
	err = kernel.setArg(1, outBuf);

	vector<uchar> output(vec.size());

	// Send Device
	cl::CommandQueue queue(context, device);
	err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(vec.size()), cl::NDRange(workGroupSize)); // Execute task
	err = queue.enqueueReadBuffer(outBuf, CL_TRUE, 0, sizeof(uchar) * output.size(), output.data()); // Copy data from global memory in GPU back into CPU memory.

	Mat image = Mat(output).reshape(3, rows);

	namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
	imshow("Display window", image);                   // Show our image inside it.

	waitKey(0);
}

// Initial OpenCL blur application 
void blurOpenCL(vector<uchar> &vec, int rows) {
	//PassFilter(vec, rows);
	
	int window_size = 10;
	int width = vec.size() / rows;
	int size = vec.size();


	setUpOpenCL();

	ifstream infile("kernel.cl");
	string src(istreambuf_iterator<char>(infile), (istreambuf_iterator<char>()));
	cl::Program::Sources sources(1, make_pair(src.c_str(), src.length()));

	cl_int err = 0;

	cl::Context context(device, 0, 0, 0, &err);

	cl::Program program(context, sources);

	err = program.build("-cl-std=CL1.2");

	err = 0;
	cl::Kernel kernel(program, "average_blur", &err);
	auto workGroupSize = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
	auto workGroups = vec.size() / workGroupSize;


	cl::Buffer inBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(uchar) * vec.size(), vec.data(), &err);
	cl::Buffer outBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(uchar)* vec.size());


	err = kernel.setArg(0, inBuf);
	err = kernel.setArg(1, outBuf);
	err = kernel.setArg(2, window_size);
	err = kernel.setArg(3, width);
	err = kernel.setArg(4, size);

	vector<uchar> output(vec.size());

	// Send Device
	cl::CommandQueue queue(context, device);
	err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(vec.size()), cl::NDRange(workGroupSize)); // Execute task
	err = queue.enqueueReadBuffer(outBuf, CL_TRUE, 0, sizeof(uchar) * output.size(), output.data()); // Copy data from global memory in GPU back into CPU memory.

	Mat image = Mat(output).reshape(3, rows);

	namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
	imshow("Display window", image);                   // Show our image inside it.

	waitKey(0);

	cout << "ok" << endl;
}

// OpenCL with better data structure
void blurImageProcessDataStructure() {

}

// Start of comparision of OpenCV and OpenCL process.
void blurImageProcess() {
	Mat image;
	vector<uchar> imageVec;
	stringstream inputImage;
	string currentImage;
	
	for (const auto & entry : directory_iterator(currentDir + "/input")) {
		inputImage << entry.path() << endl;

		currentImage = inputImage.str();

		currentImage = currentImage.substr(1, currentImage.find_last_of('"') - 1);
		image = imread(currentImage, IMREAD_COLOR);

		if (!image.data) {
			cout << "Could not open or find the image" << endl;
			break;
		}

		//blurOpenCV(image);

		imageVec.assign(image.datastart, image.dataend);
		blurOpenCL(imageVec, image.rows);

		inputImage.str(string());
		inputImage.clear();
	}

	cout << "Time (OpenCV): " << OpenCVTime<< " seconds" << endl;
}


int main() {
	currentDir = getCurrentDir();

	blurImageProcess();

	
	//OpenCLTest();
	//OpenCVTest();
}