__kernel void NumericalReduction(__global int* data, __local int* localData, __global int* outData) {
	size_t global_id = get_global_id(0);
	size_t local_size = get_global_size(0);
	size_t local_id = get_local_id(0);

	localData[local_id] = data[global_id];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = local_size >> 1; i > 0; i >>= 1) {
		if (local_id < i) {
			localData[local_id] += localData[local_id + i];
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (local_id == 0) {
		// ID of work group
		outData[get_group_id(0)] = localData[0];
	}
}

__kernel void pass_filter(const __global  uchar* input, __global uchar* output) {
	int index = get_global_id(0);

	if (index % 3 == 2) {
		output[index] = input[index];
	}
}


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

__kernel void average_blur_data(const __global  int3* input, __global int3* output, int half_size, int width, int size) {
	int index = get_global_id(0);

	int relative_index = index % width;
	int area = 0;
	int3 current;

	for (int r = -half_size; r <= half_size; r++)
	{
		if (index + r * width >= 0 && index + r * width < size) {
			for (int c = -half_size; c <= half_size; c++) {
				if (relative_index + c >= 0 && relative_index + c < width) {
					current += input[index + r * width + c];
					area++;
				}
			}
		}
	}

	output[index] = current / area;
}

