/* * TorchKissAnn - A collection of tools to train various types of Machine learning
 * algorithms on various types of EIS data
 * Copyright (C) 2025 Carl Klemm <carl@uvos.xyz>
 *
 * This file is part of TorchKissAnn.
 *
 * TorchKissAnn is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * TorchKissAnn is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with TorchKissAnn.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once
#include <cstdint>
#include <limits>
#include <string>
#include <map>

#include "net.h"
#include "tensoroptions.h"
#include "indicators.hpp"
#include "randomgen.h"

struct DropDesc
{
	bool dropout;
	float max;
	float min;
	float strength;
};

template <typename DataSelf>
class EisDataset:
public torch::data::datasets::Dataset<DataSelf, torch::data::Example<torch::Tensor, torch::Tensor>>
{
private:
	std::map<int64_t, int64_t> labelMap;
	std::vector<DropDesc> dropouts;

protected:
	virtual torch::data::Example<torch::Tensor, torch::Tensor> getImpl(size_t index) = 0;
	virtual torch::Tensor getTargetImpl(size_t index);

public:
	torch::data::Example<torch::Tensor, torch::Tensor> get(size_t index) override;
	torch::Tensor getTarget(size_t index);
	virtual size_t outputSize() const = 0;
	virtual std::string outputName(size_t output);
	virtual c10::optional<size_t> size() const override = 0;
	virtual bool isMulticlass();
	bool createLabelMap(const ann::Net& net);
	void setDropouts(const std::vector<DropDesc>& dropouts);
	const std::vector<DropDesc>& getDropouts();

	virtual torch::Tensor classCounts();
	virtual torch::Tensor classWeights();
	virtual size_t inputSize();
	virtual c10::optional<torch::Tensor> frequencies();
	virtual std::string dataLabel() const;
	virtual std::vector<std::pair<std::string, int64_t>> extraInputs();
	virtual std::pair<torch::Tensor, torch::Tensor> inputRanges();

	virtual ~EisDataset() = default;
};

template <typename DataSelf>
std::string EisDataset<DataSelf>::dataLabel() const
{
	return "EIS";
}

template <typename DataSelf>
std::string EisDataset<DataSelf>::outputName(size_t output)
{
	return std::string("Unkown");
}

template <typename DataSelf>
bool EisDataset<DataSelf>::isMulticlass()
{
	return false;
}

template <typename DataSelf>
torch::Tensor EisDataset<DataSelf>::getTarget(size_t index)
{
	torch::Tensor target = getTargetImpl(index);
	if(!isMulticlass())
	{
		auto search = labelMap.find(target[0].item().to<int64_t>());
		if(search != labelMap.end())
			target[0] = search->second;
	}
	return target;
}

template <typename DataSelf>
torch::Tensor EisDataset<DataSelf>::getTargetImpl(size_t index)
{
	return getImpl(index).target;
}

template <typename DataSelf>
bool EisDataset<DataSelf>::createLabelMap(const ann::Net& net)
{
	if(outputSize() > net.getOutputLabels().size())
		return false;

	for(int64_t i = 0; i < outputSize(); ++i)
	{
		std::string model = outputName(i);
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

	if(labelMap.size() != outputSize())
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

	if(!dropouts.empty())
	{
		for(size_t i = 0; i < dropouts.size(); ++i)
		{
			if(dropouts[i].dropout)
			{
				float random = rd::rand(dropouts[i].min, dropouts[i].max);
				data.data[i] = data.data[i]*(1-dropouts[i].strength) + random*dropouts[i].strength;
			}
		}
	}

	return data;
}

template <typename DataSelf>
torch::Tensor EisDataset<DataSelf>::classCounts()
{
	assert(outputSize() > 0);
	torch::TensorOptions options;
	options = options.dtype(torch::kFloat32);
	options = options.layout(torch::kStrided);
	options = options.device(torch::kCPU);
	return torch::ones({static_cast<int64_t>(outputSize())}, options);
}

template <typename DataSelf>
torch::Tensor EisDataset<DataSelf>::classWeights()
{
	torch::Tensor output = classCounts();
	int64_t count = size().value();

	output = output/static_cast<double>(count);
	output = torch::min(output)/output;
	return output;
}

template <typename DataSelf>
c10::optional<torch::Tensor> EisDataset<DataSelf>::frequencies()
{
	return c10::optional<torch::Tensor>();
}

template <typename DataSelf>
size_t EisDataset<DataSelf>::inputSize()
{
	torch::data::Example<torch::Tensor, torch::Tensor> example = getImpl(0);
	return example.data.numel();
}

template <typename DataSelf>
std::vector<std::pair<std::string, int64_t>> EisDataset<DataSelf>::extraInputs()
{
	return std::vector<std::pair<std::string, int64_t>>();
}

class dataset_error: public std::exception
{
	std::string whatStr;
public:
	dataset_error(const std::string& whatIn): whatStr(whatIn)
	{}
	virtual const char* what() const noexcept override
	{
		return whatStr.c_str();
	}
};

template <typename DataSelf>
std::pair<torch::Tensor, torch::Tensor> EisDataset<DataSelf>::inputRanges()
{
	torch::Tensor min = torch::full({static_cast<int64_t>(inputSize())}, std::numeric_limits<float>::max(), tensorOptCpu<float>());
	torch::Tensor max = torch::full({static_cast<int64_t>(inputSize())}, std::numeric_limits<float>::lowest(), tensorOptCpu<float>());

	torch::data::DataLoaderOptions options;
	options = options.batch_size(batch_size).workers(16);
	auto dataLoader = torch::data::make_data_loader(this->map(torch::data::transforms::Stack<>()), options);

	indicators::BlockProgressBar bar(
		indicators::option::BarWidth(50),
		indicators::option::PrefixText("Computing dataset ranges: "),
		indicators::option::ShowElapsedTime(true),
		indicators::option::ShowRemainingTime(true),
		indicators::option::MaxProgress(size().value()/batch_size)
	);

	indicators::show_console_cursor(false);

	for(const auto& batch : *dataLoader)
	{
		bar.tick();

		torch::Tensor inputs = batch.data;
		max = torch::maximum(std::get<0>(inputs.max(0)), max);
		min = torch::minimum(std::get<0>(inputs.min(0)), min);
	}

	bar.mark_as_completed();
	indicators::show_console_cursor(true);
	return {max, min};
}

template <typename DataSelf>
void EisDataset<DataSelf>::setDropouts(const std::vector<DropDesc>& dropouts)
{
	assert(dropouts.size() == inputSize());
	this->dropouts = dropouts;
}

template <typename DataSelf>
const std::vector<DropDesc >& EisDataset<DataSelf>::getDropouts()
{
	return dropouts;
}
