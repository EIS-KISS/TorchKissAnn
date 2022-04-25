#include "eisgeneratorloader.h"

#include <cstddef>
#include <vector>
#include <assert.h>

EisGeneratorDataset::EisGeneratorDataset(): omega(0, 1e6, 25)
{
	models.push_back(Model("w{20e3}p{1e-7, 0.9}"));
	std::vector<Model::Range> modelRange;
	modelRange.push_back(Model::Range(1e3, 50e3, 100));
	modelRange.push_back(Model::Range(1e-7, 20e-7, 100));
	modelRange.push_back(Model::Range(0.7, 1.2, 100));
	modelRanges.push_back(modelRange);
	modelSweepSizes.push_back(
		Model::getRequiredStepsForSweeps(modelRange));

	models.push_back(Model("c{20e-9}r{1e6}"));
	modelRange.clear();
	modelRange.push_back(Model::Range(1e-6, 1e-5, 100));
	modelRange.push_back(Model::Range(1e3, 20e3, 100));
	modelRanges.push_back(modelRange);
	modelSweepSizes.push_back(
		Model::getRequiredStepsForSweeps(modelRange));

	models.push_back(Model("r{1e6}-c{20e-9}r{1e6}"));
	modelRange.clear();
	modelRange.push_back(Model::Range(1e3, 20e3, 100));
	modelRange.push_back(Model::Range(1e-6, 1e-5, 100));
	modelRange.push_back(Model::Range(1e3, 20e3, 100));
	modelRanges.push_back(modelRange);
	modelSweepSizes.push_back(
	Model::getRequiredStepsForSweeps(modelRange));
}

torch::data::Example<torch::Tensor, torch::Tensor> EisGeneratorDataset::get(size_t index)
{
	std::vector<Model::DataPoint> data;
	size_t model = 0;
	for(; models.size(); ++model)
	{
		if(index < modelSweepSizes[model])
		{
			data = models[model].executeParamByIndex(modelRanges[model], omega, index);
			break;
		}
		else
		{
			index -= modelSweepSizes[model];
		}
	}

	{
		std::cout<<"Compnants: for index "<<index<<'\n';
		for(Componant* componant : models[model].getFlatComponants())
		{
			std::cout<<Componant::getComponantChar(componant)<<"{";
			for(size_t i = 0; i < componant->paramCount(); ++i)
			{
				std::cout<<componant->getParam()[i];
				if(i != componant->paramCount()-1)
					std::cout<<", ";
			}
			std::cout<<"}\n";
		}
	}

	assert(data.size());

	torch::TensorOptions options;
	options = options.dtype(torch::kFloat32);
	options = options.layout(torch::kStrided);
	options = options.device(torch::kCPU);
	torch::Tensor input = torch::empty({static_cast<long int>(data.size()*2)}, options);

	for(size_t i = 0; i < data.size()*2; ++i)
		input[i] = i < data.size() ? data[i].im.real() : data[i - data.size()].im.imag();

	torch::Tensor output = torch::zeros({static_cast<long int>(models.size())}, options);
	output[model] = 1;


	//torch::Tensor output = torch::Scalar(0);
	//output[0] = static_cast<float>(model);

	return torch::data::Example<torch::Tensor, torch::Tensor>(input, output);
}

size_t EisGeneratorDataset::modelCount()
{
	return models.size();
}

size_t EisGeneratorDataset::frequencies()
{
	return omega.count;
}

c10::optional<size_t> EisGeneratorDataset::size() const
{
	size_t size = 0;
	for(size_t i = 0; i < models.size(); ++i)
		size += modelSweepSizes[i];
	return size;
}
