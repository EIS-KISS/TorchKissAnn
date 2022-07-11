#include "eistotorch.h"
#include <cassert>
#include <cmath>
#include <cstdint>

torch::Tensor eisToTorch(const std::vector<eis::DataPoint>& data, torch::Tensor* freqs)
{
	torch::TensorOptions options;
	options = options.dtype(torch::kFloat32);
	options = options.layout(torch::kStrided);
	options = options.device(torch::kCPU);
	torch::Tensor input = torch::empty({static_cast<long int>(data.size()*2)}, options);
	if(freqs)
		*freqs = torch::empty({static_cast<long int>(data.size()*2)}, options);

	float* tensorDataPtr = input.contiguous().data_ptr<float>();
	float* tensorFreqDataPtr = freqs ? freqs->contiguous().data_ptr<float>() : nullptr;

	for(size_t i = 0; i < data.size()*2; ++i)
	{
		float datapoint = i < data.size() ? data[i].im.real() : data[i - data.size()].im.imag();
		if(std::isnan(datapoint) || std::isinf(datapoint))
			datapoint = 0;
		tensorDataPtr[i] = datapoint;
		if(tensorFreqDataPtr)
			tensorFreqDataPtr[i] = data[i % data.size()].omega;
	}

	return input;
}

std::vector<eis::DataPoint> torchToEis(const torch::Tensor& input)
{
	assert(input.numel() % 2 == 0);
	input.reshape({1, input.numel()});

	std::vector<eis::DataPoint> output(input.numel()/2);

	float* tensorDataPtr = input.contiguous().data_ptr<float>();

	for(int64_t i = 0; i < input.numel()/2; ++i)
	{
		output[i].omega = i;
		output[i].im.real(tensorDataPtr[i]);
		output[i].im.imag(tensorDataPtr[i+input.numel()/2]);
	}

	return output;
}
