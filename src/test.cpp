#include <asm-generic/errno.h>
#include <iostream>
#include <eisgenerator/model.h>
#include <libsvm/svm.h>
#include <random>
#include <torch/optim.h>

#include "data/generators/eisgendata.h"
#include "log.h"
#include "gan/networks.h"
#include "constants.h"

void testLoader(EisGeneratorDataset* data)
{
	torch::data::DataLoaderOptions options;
	options = options.batch_size(100);
	options = options.workers(16);
	options = options.max_jobs(32);

	std::cout<<options.batch_size()<<' '<<options.workers()<<' '<<options.max_jobs().value()<<' '<<data->is_stateful<<'\n';
	auto dataLoader = torch::data::make_data_loader(data->map(torch::data::transforms::Stack<>()), options);
	size_t i = 0;
	for(auto& batch : *dataLoader)
	{
		std::cout<<"batchnum: "<<++i<<" size "<<batch.data.numel()<<std::endl;
	}
}

void testGan(size_t size = 10)
{
	std::random_device rd;
	std::default_random_engine rande(rd());
	EisGeneratorDataset* eisData = EisGeneratorDataset::getNeisGeneratedDataset(size);
	GANGenerator gen(10, size);
	GANDiscriminator dis(10, size);
	torch::nn::BCEWithLogitsLoss lossFn(torch::nn::BCEWithLogitsLossOptions().reduction(torch::kMean));
	torch::Tensor equal = torch::full({1}, /*value=*/0.0, torch::TensorOptions().dtype(torch::kFloat));
	torch::Tensor label = torch::full({1}, /*value=*/1.0, torch::TensorOptions().dtype(torch::kFloat));
	float ideal = lossFn(equal, label).item<float>();
	std::cout<<"ideal: "<<ideal<<'\n';
	torch::optim::Adam optimizerGen(gen->parameters(), torch::optim::AdamOptions(1e-3).betas({0.5, 0.99}));
	torch::optim::Adam optimizerDis(dis->parameters(), torch::optim::AdamOptions(2.5e-4).betas({0.5, 0.99}));

	for(size_t i = 0; i < 2; ++i)
	{
		gen->train();
		dis->train();

		torch::Tensor inputLabel = torch::full({1}, 1, torch::TensorOptions().dtype(torch::kFloat)).to(DATA_LOCATION);
		torch::Tensor input = eisData->get(rande() % eisData->size().value()).data.reshape({10,1});

		torch::Tensor fakeLabel = torch::full({1}, 0, torch::TensorOptions().dtype(torch::kFloat)).to(DATA_LOCATION);
		torch::Tensor generatedFake = gen->forward(torch::randn({10,512}));

		torch::Tensor fakePrediction = dis->forward(generatedFake).first.view({-1});
		std::cout<<"generatedFake: "<<generatedFake.sizes()<<std::endl;
		std::cout<<"fakePrediction: "<<fakePrediction.sizes()<<std::endl;
		torch::Tensor realPrediction = dis->forward(input).first.view({-1});
		std::cout<<"realPrediction:\n"<<realPrediction<<std::endl;
		exit(0);
		torch::Tensor fakeLoss = lossFn(fakePrediction, fakeLabel);
		torch::Tensor realLoss = lossFn(realPrediction, inputLabel);
		torch::Tensor disLoss = realLoss + fakeLoss;
		if(disLoss.item<float>() > ideal * 2.0 * 0.5)
		{
			optimizerDis.zero_grad();
			disLoss.backward();
			optimizerDis.step();
		}

		fakePrediction = dis->forward(generatedFake).first.view({-1});
		torch::Tensor genLoss = lossFn(fakePrediction, inputLabel);
		if(genLoss.item<float>() > ideal * 0.5)
		{
			optimizerGen.zero_grad();
			genLoss.backward();
			optimizerGen.step();
		}
		Log(Log::INFO)<<"iter "<<i;
	}
}

int main(int argc, char** argv)
{
	Log::level = Log::DEBUG;

	std::cout<<std::setprecision(3)<<std::fixed<<std::setw(3);

	testGan();

}
