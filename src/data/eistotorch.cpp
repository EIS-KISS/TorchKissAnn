#include "eistotorch.h"
#include <cassert>
#include <cmath>
#include <cstdint>

#include "tensoroptions.h"

torch::Tensor eisToComplexTensor(const std::vector<eis::DataPoint>& data, torch::Tensor* freqs)
{
	torch::TensorOptions options = tensorOptCpu<fvalue>();

	if constexpr(std::is_same<fvalue, float>::value)
		options = options.dtype(torch::kComplexFloat);
	else
		options = options.dtype(torch::kComplexDouble);
	torch::Tensor output = torch::empty({static_cast<long int>(data.size())}, options);
	if(freqs)
		*freqs = torch::empty({static_cast<long int>(data.size())}, tensorOptCpu<fvalue>());

	torch::Tensor real = torch::real(output);
	torch::Tensor imag = torch::imag(output);

	auto realAccessor = real.accessor<fvalue, 1>();
	auto imagAccessor = imag.accessor<fvalue, 1>();
	float* tensorFreqDataPtr = freqs ? freqs->contiguous().data_ptr<float>() : nullptr;

	for(size_t i = 0; i < data.size(); ++i)
	{
		fvalue real = data[i].im.real();
		fvalue imag = data[i].im.imag();
		if(std::isnan(real) || std::isinf(real))
			real = 0;
		if(std::isnan(imag) || std::isinf(imag))
			real = 0;

		realAccessor[i] = real;
		imagAccessor[i] = imag;
		if(tensorFreqDataPtr)
			tensorFreqDataPtr[i] = data[i % data.size()].omega;
	}

	return output;
}

torch::Tensor eisToTorch(const std::vector<eis::DataPoint>& data, torch::Tensor* freqs)
{
	torch::Tensor input = torch::empty({static_cast<long int>(data.size()*2)}, tensorOptCpu<fvalue>());
	if(freqs)
		*freqs = torch::empty({static_cast<long int>(data.size()*2)}, tensorOptCpu<fvalue>());

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

torch::Tensor fvalueVectorToTensor(std::vector<fvalue>& vect)
{
	return torch::from_blob(vect.data(), {static_cast<int64_t>(vect.size())}, tensorOptCpu<fvalue>());
}
