#pragma once

#include "../data/generators/eisgendata.h"

class NearestBase
{
protected:
	NearestBase();
	torch::Tensor distM(torch::Tensor a, torch::Tensor b, int power = 2);

};

class NearestNeighbor : public NearestBase
{
	int _power;
	torch::Tensor _data;
	torch::Tensor _labels;

public:
	NearestNeighbor(torch::Tensor data, torch::Tensor lables, int power = 2);
	NearestNeighbor(int power = 2);
	void train(torch::Tensor data, torch::Tensor lables);
	torch::Tensor predict(const torch::Tensor& input);

	static torch::Tensor use(torch::Tensor input, EisGeneratorDataset* dataset)
	{
		input = input.reshape({1, input.size(0)});
		torch::data::DataLoaderOptions options;
		options = options.batch_size(dataset->size().value()).workers(16);
		auto dataLoader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(dataset->map(torch::data::transforms::Stack<>()), options);
		torch::data::Example<torch::Tensor, torch::Tensor> grandBatch = *dataLoader->begin();
		NearestNeighbor nearest(grandBatch.data, grandBatch.target);
		size_t classNum = nearest.predict(input)[0].item().to<float>();
		torch::Tensor output = torch::zeros(dataset->modelCount());
		output[classNum] = 1.0;
		return output;
	}
};
