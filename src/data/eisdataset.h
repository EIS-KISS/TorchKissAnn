#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <map>
#include "../ann/net.h"

template <typename DataSelf>
class EisDataset:
public torch::data::datasets::Dataset<DataSelf, torch::data::Example<torch::Tensor, torch::Tensor>>
{
private:
	std::map<int64_t, int64_t> labelMap;
	virtual torch::data::Example<torch::Tensor, torch::Tensor> getImpl(size_t index) = 0;

public:
	torch::data::Example<torch::Tensor, torch::Tensor> get(size_t index);
	virtual size_t classesCount() const = 0;
	virtual c10::optional<size_t> size() const override = 0;
	virtual size_t classForIndex(size_t index) = 0;
	virtual std::string modelStringForClass(size_t classNum);
	virtual bool isMulticlass();
	bool createLabelMap(const ann::Net& net);
	virtual torch::Tensor classCounts();
	virtual torch::Tensor classWeights();
};

template <typename DataSelf>
std::string EisDataset<DataSelf>::modelStringForClass(size_t classNum)
{
	return std::string("Unkown");
}

template <typename DataSelf>
bool EisDataset<DataSelf>::isMulticlass()
{
	return false;
}

template <typename DataSelf>
bool EisDataset<DataSelf>::createLabelMap(const ann::Net& net)
{
	if(classesCount() > net.getOutputLabels().size())
		return false;

	for(int64_t i = 0; i < classesCount(); ++i)
	{
		std::string model = modelStringForClass(i);
		for(int64_t j = 0; j < net.getOutputLabels().size(); ++j)
		{
			if(net.getOutputLabels()[j] == model)
			{
				std::pair<std::map<int64_t, int64_t>::iterator, bool> ret = labelMap.insert(i, j);
				if(!ret.second)
				{
					labelMap.clear();
					return false;
				}
			}
		}
	}

	if(labelMap.size() != classesCount())
		return false;
	return true;
}

template <typename DataSelf>
torch::data::Example<torch::Tensor, torch::Tensor> EisDataset<DataSelf>::get(size_t index)
{
	torch::data::Example<torch::Tensor, torch::Tensor> data = getImpl(index);
	if(!isMulticlass())
	{
		auto search = labelMap.find(data.target[0].item().to<int64_t>());
		if(search != labelMap.end())
			data.target[0] = search->second;
	}

	return data;
}

template <typename DataSelf>
torch::Tensor EisDataset<DataSelf>::classCounts()
{
	torch::TensorOptions options;
	options = options.dtype(torch::kFloat32);
	options = options.layout(torch::kStrided);
	options = options.device(torch::kCPU);
	return torch::zeros({static_cast<int64_t>(classesCount())}, options);
}

template <typename DataSelf>
torch::Tensor EisDataset<DataSelf>::classWeights()
{
	torch::Tensor output = classCounts();
	int64_t count = size().value();

	output = output/static_cast<double>(count);
	output = torch::min(output)/output;

	//output = torch::nn::functional::normalize(output, torch::nn::functional::NormalizeFuncOptions().p(2).dim(0));
	return output;
}
