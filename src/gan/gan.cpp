#include "gan.h"

#include <cstdint>
#include <torch/optim.h>

#include "log.h"
#include "simplenet.h"
#include "constants.h"

static constexpr int Z_SIZE = 10;

using namespace gan;

void gan::train(std::string fileName, int sweepDepth, double noise)
{
	std::cout<<std::setprecision(10)<<std::fixed<<std::setw(5);
	EisGeneratorDataset* eisData = EisGeneratorDataset::getNeisGeneratedDataset(DATA_WIDTH, sweepDepth, noise, true);

	torch::data::DataLoaderOptions options;
	options = options.batch_size(100);
	options = options.workers(16);
	options = options.max_jobs(32);
	auto dataLoader = torch::data::make_data_loader(eisData->map(torch::data::transforms::Stack<>()), options);

	GANGenerator gen(Z_SIZE, DATA_WIDTH);
	GANDiscriminator dis(DATA_WIDTH);

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

			torch::Tensor inputLabel = torch::full({batch.data.size(0)}, 1, torch::TensorOptions().dtype(torch::kFloat)).to(DATA_LOCATION);
			torch::Tensor input = batch.data;

			torch::Tensor fakeLabel = torch::full({batch.data.size(0)}, 0, torch::TensorOptions().dtype(torch::kFloat)).to(DATA_LOCATION);
			torch::Tensor generatedFake = gen->forward(torch::randn({batch.data.size(0), Z_SIZE}));

			torch::Tensor fakePrediction = dis->forward(generatedFake.detach()).first.view({-1});

			torch::Tensor realPrediction = dis->forward(input).first.view({-1});
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

			torch::Tensor accReal = realPrediction.signbit().eq(fakeLabel).sum();
			torch::Tensor accFake = fakePrediction.signbit().eq(inputLabel).sum();
			/*std::cout<<"accReal: "<<accReal.item<double>()<<" accFake: "<<accFake.item<double>()<<'\n';*/

			if(index++ % 1000 == 0)
			{
				std::cout<<"realPrediction:\n"<<realPrediction<<'\n';
				std::cout<<"fakePrediction:\n"<<fakePrediction<<'\n';
				size_t data_size = eisData->size().value();
				int64_t end = std::min(data_size, (index + 1) * BATCH_SIZE);
				std::cout<<"Train Epoch: "<< epoch <<' '<<end<<'/'<<data_size<<"\tDis Loss: "<<(disLoss/torch::Scalar(end)).item<float>()<<"\tGen Loss: "<<(genLoss/torch::Scalar(end)).item<float>()<<"\tAcc: "<<(accReal+accFake).item<float>()<<'\n';
				torch::save(dis, fileName + ".dis.vos.uf");
				torch::save(gen, fileName + ".gen.vos.uf");
			}
		}
	}
	torch::save(dis, fileName + ".dis.vos");
	torch::save(gen, fileName + ".gen.vos");
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> anomalyScore(torch::Tensor data, torch::Tensor fake, GANDiscriminator &dis, float lambda = 0.1)
{
    torch::Tensor resLoss = torch::abs(data - fake).sum();
    torch::Tensor feature = dis->forward(data).second;
    torch::Tensor fakeFeature = dis->forward(fake).second;
    torch::Tensor disLoss = torch::abs(feature - fakeFeature).sum();
    torch::Tensor score = (1.0 - lambda) * resLoss + lambda * disLoss;
    return {score, resLoss, disLoss};
}

torch::Tensor gan::generate(const std::string& fileName)
{
	GANGenerator gen(Z_SIZE, DATA_WIDTH);
	torch::load(gen, fileName + ".gen.vos");
	return gen->forward(torch::randn({1, Z_SIZE}));
}

bool gan::filter(torch::Tensor input, const std::string& fileName, float nabla)
{
	GANGenerator gen(Z_SIZE, DATA_WIDTH);
	GANDiscriminator dis(DATA_WIDTH);
	torch::load(dis, fileName + std::string(".dis.vos"));
	torch::load(gen, fileName + std::string(".gen.vos"));
	dis->eval();
	gen->eval();

	torch::Tensor fake = gen->forward(torch::randn({1, Z_SIZE}));

	auto [score, resLoss, disLoss] = anomalyScore(input, fake, dis);
	Log(Log::DEBUG)<<"score: "<<score.item<float>();
	return score.item<float>() < nabla;
}
