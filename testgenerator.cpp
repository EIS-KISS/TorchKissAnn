#include "testgenerator.h"
#include <cstddef>
#include <vector>
#include <assert.h>
#include <cmath>

TestDataset::TestDataset()
{
}

torch::data::Example<torch::Tensor, torch::Tensor> TestDataset::get(size_t index)
{
	torch::TensorOptions options;
	options = options.dtype(torch::kFloat32);
	options = options.layout(torch::kStrided);
	options = options.device(torch::kCPU);
	torch::Tensor input = torch::zeros({static_cast<long int>(OUTPUT_SIZE)}, options);

	size_t classNum = index % COUNT_CLASSES;

	for(size_t i = 0; i < OUTPUT_SIZE; ++i)
	{
		float x = COUNT_CLASSES*(static_cast<float>(i)+((static_cast<float>(classNum)/COUNT_CLASSES)*OUTPUT_SIZE))/OUTPUT_SIZE;
		input[i] = 1/(std::pow(x , 2)+1);
	}

	options = options.dtype(torch::kInt64);
	torch::Tensor output = torch::empty({1}, options);;
	output[0] = static_cast<float>(classNum);

	return torch::data::Example<torch::Tensor, torch::Tensor>(input, output);
}

c10::optional<size_t> TestDataset::size() const
{
	return 1e6;
}
