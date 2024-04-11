#pragma once
#include <torch/optim.h>

#include "trainlog.h"
#include "simplenet.h"
#include "../data/eisdataset.h"
#include "globals.h"

namespace gan
{
	static constexpr int Z_SIZE = 10;

	bool filter(torch::Tensor input, const std::string& fileName, float nabla = 20);
	torch::Tensor generate(const std::string& fileName);

	template <typename DatasetType>
	void train(TrainLog* trainLog, EisDataset<DatasetType>* trainDataset, size_t epochs, double learingRate, double* finalLoss = nullptr)
	{
		torch::data::DataLoaderOptions options;
		options = options.batch_size(100);
		options = options.workers(16);
		options = options.max_jobs(32);
		auto dataLoader = torch::data::make_data_loader(trainDataset->map(torch::data::transforms::Stack<>()), options);

		size_t data_size = trainDataset->size().value();
		int64_t loginterval = data_size/10000 ?: data_size/batch_size-1;

		GANGenerator gen(Z_SIZE, trainDataset->get(0).data.size(0));
		GANDiscriminator dis(trainDataset->get(0).data.size(0));

		torch::nn::BCEWithLogitsLoss lossFn(torch::nn::BCEWithLogitsLossOptions().reduction(torch::kMean));
		torch::Tensor equal = torch::full({1}, /*value=*/0.0, torch::TensorOptions().dtype(torch::kFloat));
		torch::Tensor label = torch::full({1}, /*value=*/1.0, torch::TensorOptions().dtype(torch::kFloat));

		float ideal = lossFn(equal, label).item<float>();
		Log(Log::INFO)<<"ideal: "<<ideal;

		torch::optim::Adam optimizerGen(gen->parameters(), torch::optim::AdamOptions(1e-3).betas({0.5, 0.99}));
		torch::optim::Adam optimizerDis(dis->parameters(), torch::optim::AdamOptions(2.5e-4).betas({0.5, 0.99}));

		for(size_t epoch = 0; epoch < 1; ++epoch)
		{
			size_t index = 0;
			gen->train();
			dis->train();

			for(auto& batch : *dataLoader)
			{

				torch::Tensor inputLabel = torch::full({batch.data.size(0)}, 1, torch::TensorOptions().dtype(torch::kFloat)).to(*offload_device);
				torch::Tensor input = batch.data;

				torch::Tensor fakeLabel = torch::full({batch.data.size(0)}, 0, torch::TensorOptions().dtype(torch::kFloat)).to(*offload_device);
				torch::Tensor generatedFake = gen->forward(torch::randn({batch.data.size(0), Z_SIZE}));

				torch::Tensor fakePrediction = dis->forwardSplit(generatedFake.detach()).first.view({-1});

				torch::Tensor realPrediction = dis->forwardSplit(input).first.view({-1});
				torch::Tensor fakeLoss = lossFn(fakePrediction, fakeLabel);
				torch::Tensor realLoss = lossFn(realPrediction, inputLabel);
				torch::Tensor disLoss = realLoss + fakeLoss;
				if(disLoss.item<float>() > ideal * 2.0 * 0.5)
				{
					optimizerDis.zero_grad();
					disLoss.backward();
					optimizerDis.step();
				}

				fakePrediction = dis->forwardSplit(generatedFake).first.view({-1});
				torch::Tensor genLoss = lossFn(fakePrediction, inputLabel);
				if(genLoss.item<float>() > ideal * 0.5)
				{
					optimizerGen.zero_grad();
					genLoss.backward();
					optimizerGen.step();
				}

				torch::Tensor accReal = realPrediction.signbit().eq(fakeLabel).sum();
				torch::Tensor accFake = fakePrediction.signbit().eq(inputLabel).sum();
				/*std::cout<<"accReal: "<<accReal.item<double>()<<" accFake: "<<accFake.item<double>()<<'\n';*/

				if(trainLog && index % loginterval == 0)
				{
					auto end = std::min(data_size, (index + 1) * batch_size);
					trainLog->logTrainLoss(epoch, end, disLoss.item<float>(), accReal.item<double>(), data_size);
				}
			}
		}
		//trainLog->saveNetwork();
		/*torch::save(dis, fileName + ".dis.vos");
		torch::save(gen, fileName + ".gen.vos");*/
	}
}
