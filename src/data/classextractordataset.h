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
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <optional>
#include <mutex>
#include <memory>

#include "data/eisdataset.h"

template <typename ParentType>
class ClassExtractorDataset : public EisDataset<ClassExtractorDataset<ParentType>>
{
private:
	EisDataset<ParentType>* _dataset;
	std::shared_ptr<std::mutex> datasetMutex;
	size_t _classNum;

	std::vector<size_t> indexList;

	virtual torch::data::Example<torch::Tensor, torch::Tensor> getImpl(size_t index) override
	{
		assert(index < indexList.size());

		datasetMutex->lock();
		torch::data::Example<torch::Tensor, torch::Tensor> example = _dataset->get(indexList[index]);
		datasetMutex->unlock();
		return example;
	}

public:

	ClassExtractorDataset(EisDataset<ParentType>* dataset, size_t classNum):
	_dataset(dataset), datasetMutex(new std::mutex), _classNum(classNum)
	{
		size_t dataSetSize = _dataset->size().value();
		for(size_t i = 0; i < dataSetSize; ++i)
		{
			torch::Tensor target = _dataset->getTarget(i);
			if(target.item().toUInt64() == classNum)
				indexList.push_back(i);
		}
	}

	virtual size_t outputSize() const override
	{
		return 1;
	}

	virtual std::string outputName(size_t classNum) override
	{
		return _dataset->outputName(_classNum);
	}

	size_t getParentIndex(size_t index)
	{
		assert(index < indexList.size());

		return indexList[index];
	}

	virtual c10::optional<size_t> size() const override
	{
		return indexList.size();
	}
};
