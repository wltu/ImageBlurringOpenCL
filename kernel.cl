__kernel void average_blur(const __global  uchar* input, __global uchar* output, int half_size, int width, int size) {
	int index = get_global_id(0);


	if (index % 3 == 0) {
		int relative_index = index % width;

		int area = 0;
		int x = 0;
		int y = 0;
		int z = 0;
		int newIndex;

		for (int r = -half_size; r <= half_size; r++)
		{
			if (index + r * width >= 0 && index + r * width < size) {
				for (int c = -half_size; c <= half_size; c++) {

					if (relative_index + 3*c >= 0 && relative_index + 3*c < width) {
						newIndex = index + r * width + 3 * c;
						x += input[newIndex];
						y += input[newIndex + 1];
						z += input[newIndex + 2];

						area++;
					}
				}
			}
		}

		output[index] = x / area;
		output[index + 1] = y / area;
		output[index + 2] = z / area;
	}
}

__kernel void convert_data(const __global  uchar* input, __global uchar3* output) {
	int index = get_global_id(0);

	if (index % 3 == 0) {
		output[index / 3].x = input[index];
		output[index / 3].y = input[index + 1];
		output[index / 3].z = input[index + 2];
	}
}

__kernel void convert_data_back(const __global  uchar3* input, __global uchar* output) {
	int index = get_global_id(0) * 3;

	uchar3 num = input[index / 3];

	output[index] = num.x;
	output[index + 1] = num.y;
	output[index + 2] = num.z;
}

__kernel void average_blur_data(const __global  uchar3* input, __global uchar3* output, int half_size, int width, int size) {
	int index = get_global_id(0);

	int relative_index = index % width;
	int area = 0;
	int3 current;
	uchar3 p;

	for (int r = -half_size; r <= half_size; r++)
	{
		if (index + r * width >= 0 && index + r * width < size) {
			for (int c = -half_size; c <= half_size; c++) {
				if (relative_index + c >= 0 && relative_index + c < width) {
					p = input[index + r * width + c];

					current.x += p.x;
					current.y += p.y;
					current.z += p.z;
					
					area++;
				}
			}
		}
	}

	output[index].x = current.x / area;
	output[index].y = current.y / area;
	output[index].z = current.z / area;
}

