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
	virtual torch::Tensor classCounts()
	{
		torch::TensorOptions options;
		options = options.dtype(torch::kFloat32);
		options = options.layout(torch::kStrided);
		options = options.device(torch::kCPU);
		return torch::zeros({static_cast<int64_t>(classesCount())}, options);
	}
	virtual torch::Tensor classWeights()
	{
		torch::Tensor output = classCounts();
		int64_t count = size().value();

		output = output/static_cast<double>(count);
		output = torch::min(output)/output;

		//output = torch::nn::functional::normalize(output, torch::nn::functional::NormalizeFuncOptions().p(2).dim(0));
		return output;
	}
};
