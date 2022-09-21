#pragma once
#include <cstdint>
#include <optional>
#include <string>

template <typename DataSelf>
class EisDataset:
public torch::data::datasets::Dataset<DataSelf, torch::data::Example<torch::Tensor, torch::Tensor>>
{
public:
	virtual torch::data::Example<torch::Tensor, torch::Tensor> get(size_t index) override = 0;
	virtual size_t classesCount() const = 0;
	virtual c10::optional<size_t> size() const override = 0;
	virtual size_t classForIndex(size_t index) = 0;
	virtual std::string modelStringForClass(size_t classNum) {return std::string("Unkown");}
	virtual bool isMulticlass() {return false;}
	virtual torch::Tensor classWeights()
	{
		torch::TensorOptions options;
		options = options.dtype(torch::kFloat32);
		options = options.layout(torch::kStrided);
		options = options.device(torch::kCPU);
		return torch::ones({static_cast<int64_t>(classesCount())}, options);
	}
};
