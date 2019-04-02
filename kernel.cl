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
