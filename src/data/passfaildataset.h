#pragma once
#include <string>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/types.h>

#include "eisdataset.h"
#include "randomgen.h"
#include "tensoroptions.h"

template <typename SourceDataset>
class PassFaillDataset:
public EisDataset<PassFaillDataset<SourceDataset>>
{
	SourceDataset* dataset_;

	void normalize(torch::Tensor& tensor)
	{
		//tensor = tensor - torch::min(tensor);
		tensor = tensor / torch::max(tensor);
	}

	virtual torch::data::Example<torch::Tensor, torch::Tensor> getImpl(size_t index) override
	{
		torch::data::Example example = dataset_->get(index);
		example.target = torch::ones({1});
		example.data.requires_grad_(false);
		if(index < dataset_->size().value())
		{
			example.target = torch::zeros({1});
			if(rd::rand() < 0.01)
			{
				example.data = torch::randn(example.data.sizes(), tensorOptCpu<float>(false));
				normalize(example.data);
			}
			else
			{
				double magnitude = rd::rand(0.02)+0.01;
				torch::Tensor random = torch::randint(-1e8, 1e8, example.data.sizes());
				random.requires_grad_(false);
				random = (random.to(example.data.dtype())/1e8)*magnitude;
				random[0] = 0;
				random[random.numel()-1] = 0;
				example.data += random;
				normalize(example.data);
			}
		}
		return example;
	}

public:
	PassFaillDataset(SourceDataset *dataset):
	dataset_(dataset)
	{}

	virtual size_t outputSize() const override
	{
		return 2;
	}

	virtual c10::optional<size_t> size() const override
	{
		return dataset_->size().value()*2;
	}

	virtual size_t classForIndex(size_t index) override
	{
		return index > dataset_->size().value();
	}

	virtual std::string modelStringForClass(size_t classNum) override
	{
		return classNum == 0 ? "Fail" : "Pass";
	}

	virtual bool isMulticlass() override
	{
		return false;
	}
};
