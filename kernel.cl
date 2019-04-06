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

__kernel void red_filter(const __global  int* input, __global int* output) {
	int rowOffset = get_global_id(1) * get_global_size(0) * 3;
	int my = 3 * get_global_id(0) + rowOffset;

	int area = (hfs * 2) * (hfs * 2);
	int fIndex = 0;
	int sumR = 0.0;
	int sumG = 0.0;
	int sumB = 0.0;
	int offset;
	int curRow;


	output[my] = input[my];
	output[my + 1] = 0;
	output[my + 2] = 0;
}

/*
__kernel void red_filter(const __global  int* input, __global int* output, int hfs, int width) {
	int rowOffset = get_global_id(1) * IMAGE_W * 3;
	int my = 3 * get_global_id(0) + rowOffset;

	int area = (hfs * 2) * (hfs * 2);
	int fIndex = 0;
	int sumR = 0.0;
	int sumG = 0.0;
	int sumB = 0.0;
	int offset;
	int curRow;

	for (int r = -hfs; r <= hfs; r++)
	{
		curRow = my + r * (width * 3);
		for (int c = -hfs; c <= hfs; c++, fIndex += 3) {
			offset = c * 4;

			sumR += input[curRow + offset];
			sumG += input[curRow + offset + 1];
			sumB += input[curRow + offset + 2];
		}
	}

	output[my] = sumR / area;
	output[my + 1] = sumG / area;
	output[my + 2] = sumB / area;
}
*/