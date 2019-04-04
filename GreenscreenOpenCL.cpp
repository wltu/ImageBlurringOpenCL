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


string currentDir = "";
int rows = -1;

void OpenCLTest() {
	vector<cl::Platform> platforms;
	vector<cl::Device> devices;
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


void saveData(vector<vector<uchar>> &vec, string dir) {
	Mat image;
	vector<uchar> imageVec;
	stringstream inputImage;
	string currentImage;

	for (const auto & entry : directory_iterator(currentDir + dir)) {
		inputImage << entry.path() << endl;

		currentImage = inputImage.str();

		currentImage = currentImage.substr(1, currentImage.find_last_of('"') - 1);
		image = imread(currentImage, IMREAD_COLOR);

		if (rows == -1) {
			rows = image.rows;
		}
		else {
			if (rows != image.rows) {
				cout << "Image Size does not match!";

				return;
			}
		}

		if (!image.data) {
			cout << "Could not open or find the image" << endl;
			return;
		}


		imageVec.assign(image.datastart, image.dataend);
		vec.push_back(imageVec);

		inputImage.str(string());
		inputImage.clear();
	}
}

void initImage(vector<vector<uchar>> &inputsImage, vector<vector<uchar>> &inputsBackground) {
	saveData(inputsImage, "/input/greenscreen");
	saveData(inputsBackground, "/input/background");
}

void CPU_Process(vector<vector<uchar>> input, vector<vector<uchar>> background) {
	Mat image;

	vector<uchar> imageVec;
	vector<uchar> backVec;

	double start = omp_get_wtime();

	for (int k = 0; k < input.size(); k++) {
		for (int j = 0; j < background.size(); j++) {
			imageVec = input[k];
			backVec = background[j];

			if (backVec.size() == imageVec.size()) {
				for (int i = 0; i < imageVec.size(); i += 3) {
					if (imageVec[i] < 50 && imageVec[i + 2] < 50 && imageVec[i + 1] > 200) {
						imageVec[i] = backVec[i];
						imageVec[i + 1] = backVec[i + 1];
						imageVec[i + 2] = backVec[i + 2];
					}
				}
			}
		}
	}

	cout << "Time: " << omp_get_wtime() - start << " seconds" << endl;
}

void OpenCL_Process() {

}

void OpenCL_Process_DataStructure() {

}



void blurTest() {
	Mat image;

	image = imread("input/greenscreen/greenscreen.jpg", IMREAD_COLOR);

	blur(image, image, Size(10, 10));


	namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
	imshow("Display window", image);                   // Show our image inside it.

	waitKey(0);

}

int main() {
	currentDir = "";
	currentDir = getCurrentDir();

	/*vector<vector<uchar>> inputs;
	vector<vector<uchar>> background;
	initImage(inputs, background);

	CPU_Process(inputs, background);*/

	//OpenCLTest();
	//OpenCVTest();

	blurTest();
}