#include "eistotorch.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <torch/csrc/autograd/generated/variable_factories.h>

#include "tensoroptions.h"

torch::Tensor eisToComplexTensor(const std::vector<eis::DataPoint>& data, torch::Tensor* freqs)
{
	torch::TensorOptions options = tensorOptCpu<fvalue>(false);

	if constexpr(std::is_same<fvalue, float>::value)
		options = options.dtype(torch::kComplexFloat);
	else
		options = options.dtype(torch::kComplexDouble);
	torch::Tensor output = torch::empty({static_cast<long int>(data.size())}, options);
	if(freqs)
		*freqs = torch::empty({static_cast<long int>(data.size())}, tensorOptCpu<fvalue>(false));

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
	torch::Tensor input = torch::empty({static_cast<long int>(data.size()*2)}, tensorOptCpu<fvalue>(false));
	if(freqs)
		*freqs = torch::empty({static_cast<long int>(data.size()*2)}, tensorOptCpu<fvalue>(false));

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

torch::Tensor eisToTorchExtra(const std::vector<eis::DataPoint>& data, std::vector<fvalue> extraInputs)
{
	torch::Tensor extra = torch::from_blob(extraInputs.data(), {static_cast<int64_t>(extraInputs.size())}, tensorOptCpu<fvalue>(false)).clone();
	return torch::cat({eisToTorch(data), extra}, 0).detach();
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

torch::Tensor rangeToTensor(const eis::Range& range)
{
	torch::Tensor out = torch::zeros({3}, tensorOptCpu<fvalue>(false));

	out[0] = range.start;
	out[1] = range.end;
	out[2] = static_cast<fvalue>(range.count);

	return out;
}

eis::Range tensorToRange(const torch::Tensor& tensor)
{
	assert(tensor.numel() == 3);
	assert(tensor.size(0) == 3);

	eis::Range out(tensor[0].item().toDouble(), tensor[1].item().toDouble(), tensor[2].item().toDouble());
	return out;
}
