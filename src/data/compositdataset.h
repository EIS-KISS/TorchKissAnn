#include "data/eisdataset.h"
#include <cstdint>

template <typename DatasetAType, typename DatasetBType>
class CompositDataset : public EisDataset<CompositDataset<DatasetAType, DatasetBType>>
{
private:
	EisDataset<DatasetAType>* _datasetA;
	EisDataset<DatasetBType>* _datasetB;
	std::vector<std::string> classStrings;
	std::vector<int64_t> classIds;

	virtual torch::data::Example<torch::Tensor, torch::Tensor> getImpl(size_t index) override
	{
		assert(index < classIds.size());

		torch::data::Example<torch::Tensor, torch::Tensor> example ;

		if(index >= _datasetA->size().value())
			example = _datasetA->get(index);
		else
			example = _datasetB->get(index-_datasetA->size().value());

		example.target[0] = static_cast<int64_t>(classIds[index]);
		return example;
	}

	template<typename Type>
	void addDataset(EisDataset<Type>* dataset)
	{
		for(size_t i = 0; i < dataset->size().value(); ++i)
		{
			std::string model = dataset->modelStringForClass(dataset->classForIndex(i));
			std::vector<std::string>::iterator search = std::find(classStrings.begin(), classStrings.end(), model);
			if(search == classStrings.end())
			{
				classStrings.push_back(model);
				classIds.push_back(classStrings.size()-1);
			}
			else
			{
				classIds.push_back(search-classStrings.begin());
			}
		}
	}

public:

	CompositDataset(EisDataset<DatasetAType>* datasetA, EisDataset<DatasetBType>* datasetB):
	_datasetA(datasetA), _datasetB(datasetB)
	{
		assert(!datasetA->isMulticlass() && !datasetB->isMulticlass());
		classIds.reserve(_datasetA->size().value()+_datasetB->size().value());
		addDataset<DatasetAType>(_datasetA);
		addDataset<DatasetBType>(_datasetB);
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
