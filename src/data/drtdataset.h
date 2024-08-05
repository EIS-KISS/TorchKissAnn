#include <stdint.h>
#include <eisdrt/torchdrt.h>

#include "eisdataset.h"
#include "tensoroperators.h"
#include "tensoroptions.h"

template <typename SourceDataset>
class DrtDataset:
public EisDataset<DrtDataset<SourceDataset>>
{
	SourceDataset *dataset_;

	virtual torch::data::Example<torch::Tensor, torch::Tensor> getImpl(size_t index) override
	{
		torch::data::Example<torch::Tensor, torch::Tensor> example = dataset_->get(index);

		torch::Tensor impedanceSpectrum = linearToComplex(example.data);
		torch::Tensor omegaTensor = torch::logspace(1, 1e6, impedanceSpectrum.numel(), 10, tensorOptCpu<fvalue>());

		FitMetics fm;

		torch::Tensor drt = calcDrtTorch(impedanceSpectrum, omegaTensor, fm, FitParameters(1000));
		return torch::data::Example<torch::Tensor, torch::Tensor>(drt, example.target);
	}

public:
	DrtDataset(SourceDataset *dataset): dataset_(dataset) {}

	virtual size_t outputSize() const override
	{
		return dataset_->outputSize();
	}

	virtual c10::optional<size_t> size() const override
	{
		return dataset_->size();
	}

	virtual size_t classForIndex(size_t index)
	{
		return dataset_->classForIndex(index);
	}

	virtual std::string modelStringForClass(size_t classNum) override
	{
		return dataset_->modelStringForClass(classNum);
	}

	virtual bool isMulticlass() override
	{
		return dataset_->isMulticlass();
	}

	virtual torch::Tensor classWeights() override
	{
		return dataset_->classWeights();
	}
};
