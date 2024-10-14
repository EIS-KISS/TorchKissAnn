#pragma once

#include <kisstype/type.h>
#include <tensoroperators.h>

#include <vector>

torch::Tensor eisToComplexTensor(const std::vector<eis::DataPoint>& data, torch::Tensor* freqs = nullptr);
torch::Tensor eisToTorch(const std::vector<eis::DataPoint>& data, torch::Tensor* freqs = nullptr);
torch::Tensor eisToTorchExtra(const std::vector<eis::DataPoint>& data, std::vector<fvalue> extraInputs = {});
std::vector<eis::DataPoint> torchToEis(const torch::Tensor& input);
torch::Tensor fvalueVectorToTensor(std::vector<fvalue>& vect);

torch::Tensor rangeToTensor(const eis::Range& range);
eis::Range tensorToRange(const torch::Tensor& tensor);

template<typename fv>
std::pair<std::valarray<fv>, std::valarray<fv>> torchToValarray(torch::Tensor tensor)
{
	assert(tensor.numel() % 2 == 0);
	assert(checkTorchType<fv>(tensor));
	torch::Tensor work = tensor.reshape({tensor.numel()});

	std::pair<std::valarray<fv>, std::valarray<fv>> out(std::valarray<fv>(0.0, tensor.numel()/2), std::valarray<fv>(0.0, tensor.numel()/2));
	auto accessor = work.accessor<fv, 1>();
	for(int64_t i = 0; i < tensor.numel()/2; ++i)
		out.first[i] = accessor[i];
	for(int64_t i = tensor.numel()/2; i < tensor.numel(); ++i)
		out.second[i-tensor.numel()/2] = accessor[i];
	return out;
}
