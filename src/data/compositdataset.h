#include "data/eisdataset.h"
#include "log.h"
#include <cstdint>

template <typename DatasetAType, typename DatasetBType>
class CompositDataset : public EisDataset<CompositDataset<DatasetAType, DatasetBType>>
{
private:
	EisDataset<DatasetAType>* _datasetA;
	EisDataset<DatasetBType>* _datasetB;
	size_t aSize;
	size_t bSize;
	std::vector<std::string> classStrings;
	std::vector<int64_t> classIds;

	virtual torch::data::Example<torch::Tensor, torch::Tensor> getImpl(size_t index) override
	{
		assert(index < classIds.size());

		torch::data::Example<torch::Tensor, torch::Tensor> example ;

		if(index >= aSize)
			example = _datasetA->get(index);
		else
			example = _datasetB->get(index-aSize);

		example.target[0] = static_cast<int64_t>(classIds[index]);
		return example;
	}

	template<typename Type>
	size_t addDataset(EisDataset<Type>* dataset)
	{
		size_t modelsAdded = 0;
		for(size_t i = 0; i < dataset->size().value(); ++i)
		{
			std::string model = dataset->modelStringForClass(dataset->classForIndex(i));
			std::vector<std::string>::iterator search = std::find(classStrings.begin(), classStrings.end(), model);
			if(search == classStrings.end())
			{
				classStrings.push_back(model);
				classIds.push_back(classStrings.size()-1);
				++modelsAdded;
			}
			else
			{
				classIds.push_back(search-classStrings.begin());
			}
		}
		return modelsAdded;
	}

public:

	CompositDataset(EisDataset<DatasetAType>* datasetA, EisDataset<DatasetBType>* datasetB):
	_datasetA(datasetA), _datasetB(datasetB)
	{
		assert(!datasetA->isMulticlass() && !datasetB->isMulticlass());
		classIds.reserve(_datasetA->size().value()+_datasetB->size().value());
		size_t modsFomA = addDataset<DatasetAType>(_datasetA);
		size_t modsFromB = addDataset<DatasetBType>(_datasetB);
		aSize = _datasetA->size().value();
		bSize = _datasetB->size().value();

		Log(Log::INFO)<<__func__<<" Added "<<modsFomA<<" Models from dataset a and an additional "<<modsFromB<<" from dataset b";
		Log(Log::INFO)<<__func__<<" A Size: "<<aSize<<" B size: "<<bSize<<" Size: "<<size().value();
	}

	virtual size_t outputSize() const override
	{
		return classStrings.size();
	}

	virtual size_t classForIndex(size_t index) override
	{
		return classIds[index];
	}

	virtual std::string modelStringForClass(size_t classNum) override
	{
		return classStrings[classNum];
	}

	virtual c10::optional<size_t> size() const override
	{
		return classIds.size();
	}
};
