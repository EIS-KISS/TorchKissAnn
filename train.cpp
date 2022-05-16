#include <iostream>
#include <torch/torch.h>
#include <eisgenerator/model.h>
#include <memory>

#include "model.h"
#include "eisgeneratorloader.h"
#include "testgenerator.h"

static constexpr int batch_size = 10;
static constexpr auto DataLocation = torch::kCPU;
static constexpr int SWEEP_DEPTH = 10;

template <typename DataLoader>
void train(std::shared_ptr<Net> network, DataLoader& loader, torch::optim::Optimizer& optimizer, size_t epoch, size_t data_size)
{
	std::cout<<__func__<<'\n';
	size_t index = 0;
	network->train();
	float Loss = 0, Acc = 0;

	for (auto& batch : loader)
	{
		auto data = batch.data.to(DataLocation);
		auto targets = batch.target.to(DataLocation).view({-1});

		auto prediction = network->forward(data);
		auto loss = torch::nll_loss(prediction, targets);
		assert(!std::isnan(loss.template item<float>()));
		auto acc = prediction.argmax(1).eq(targets).sum();

		optimizer.zero_grad();
		loss.backward();
		optimizer.step();

		Loss += loss.template item<float>();
		Acc += acc.template item<float>();

		if (index++ % 1000 == 0)
		{
			auto end = std::min(data_size, (index + 1) * batch_size);

			std::cout<< "Train Epoch: " << epoch << " " << end << "/" << data_size
				<< "\tLoss: " << Loss / end << "\tAcc: " << Acc / end
				<< std::endl;
		}
	}
}


template <typename DataLoader>
void test(std::shared_ptr<Net> network, DataLoader& loader, size_t data_size)
{
	network->eval();
	torch::NoGradGuard no_grad;
	float Loss = 0, Acc = 0;

	for (const auto& batch : loader)
	{
		auto data = batch.data.to(DataLocation);
		auto targets = batch.target.to(DataLocation).view({-1});

		auto output = network->forward(data);
		auto loss = torch::nll_loss(output, targets);
		assert(!std::isnan(loss.template item<float>()));
		auto acc = output.argmax(1).eq(targets).sum();

		Loss += loss.template item<float>();
		Acc += acc.template item<float>();
	}

	std::cout<<"Test Loss: "<<Loss / data_size<< "\tAcc: " <<Acc / data_size<< std::endl;
}

inline EisGeneratorDataset* getNeisGeneratedDataset()
{
	EisGeneratorDataset* dataset = new EisGeneratorDataset();

	std::vector<eis::Range> modelRange;

	eis::Model bord1_1("(c{1e-6}-r{47})(c{1e-6}-r{1e3})(c{1e-6}-r{22e3})");

	modelRange.push_back(eis::Range(1e-7, 1e-5, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(10, 100e3, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e-7, 1e-5, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(10, 100e3, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e-7, 1e-5, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(10, 100e3, SWEEP_DEPTH, true));
	dataset->addModel(bord1_1, modelRange);

	eis::Model bord1_2("r{1e9}-c{1e-6}r{1e9}");
	modelRange.clear();
	modelRange.push_back(eis::Range(1e3, 1e10, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e-7, 1e-5, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e3, 1e10, SWEEP_DEPTH, true));
	dataset->addModel(bord1_2, modelRange);

	eis::Model bord1_4("r{100}c{1e-6}-r{100}c{100e-9}-r{100}c{100e-12}");
	modelRange.clear();
	modelRange.push_back(eis::Range(1e1, 1e4, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e-12, 1e-6, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e1, 1e4, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e-12, 1e-6, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e1, 1e4, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e-12, 1e-6, SWEEP_DEPTH, true));
	dataset->addModel(bord1_4, modelRange);

	eis::Model bord1_5("r{47}l{100e-6}-r{47}c{1e-6}-r{47}");
	modelRange.clear();
	modelRange.push_back(eis::Range(1e1, 1e3, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e-6, 1e-3, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e1, 1e3, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e-9, 1e-5, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e1, 1e3, SWEEP_DEPTH, true));
	dataset->addModel(bord1_5, modelRange);

	eis::Model bord2("r{1e3}c{1e-6}-r{1e3}c{1e-9}-r{47}");
	modelRange.clear();
	modelRange.push_back(eis::Range(1e1, 1e4, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e-12, 1e-5, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e1, 1e4, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e-12, 1e-5, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e1, 1e4, SWEEP_DEPTH, true));
	dataset->addModel(bord2, modelRange);

	eis::Model bord4("r{100}-r{470}c{100e-12}-r{1e3}c{1e-9}-r{100}c{1e-6}");
	modelRange.clear();
	modelRange.push_back(eis::Range(10, 1e4, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(10, 1e4, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e-12, 1e-5, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(10, 1e4, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e-12, 1e-5, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(10, 1e4, SWEEP_DEPTH, true));
	modelRange.push_back(eis::Range(1e-12, 1e-5, SWEEP_DEPTH, true));
	dataset->addModel(bord4, modelRange);

	std::cout<<__func__<<" with "<<dataset->size().value()<<" examples\n";
	return dataset;
}


inline EisGeneratorDataset* getNeisGeneratedDatasetAlt()
{
	EisGeneratorDataset* dataset = new EisGeneratorDataset();

	std::vector<eis::Range> modelRange;

	eis::Model bord1_1("(c{1e-6}-r{47})(c{1e-6}-r{1e3})(c{1e-6}-r{22e3})");
	modelRange.push_back(eis::Range(1e-7, 1e-5, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(10, 100e3, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(1e-7, 1e-5, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(10, 100e3, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(1e-7, 1e-5, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(10, 100e3, SWEEP_DEPTH/2, true));
	dataset->addModel(bord1_1, modelRange);

	eis::Model bord1_2("r{1e9}-c{1e-6}r{1e9}");
	modelRange.clear();
	modelRange.push_back(eis::Range(1e3, 1e10, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(1e-7, 1e-5, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(1e3, 1e10, SWEEP_DEPTH/2, true));
	dataset->addModel(bord1_2, modelRange);

	eis::Model bord1_4("r{100}c{1e-6}-r{100}c{100e-9}-r{100}c{100e-12}");
	modelRange.clear();
	modelRange.push_back(eis::Range(1e1, 1e4, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(1e-12, 1e-6, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(1e1, 1e4, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(1e-12, 1e-6, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(1e1, 1e4, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(1e-12, 1e-6, SWEEP_DEPTH/2, true));
	dataset->addModel(bord1_4, modelRange);

	eis::Model bord1_5("r{47}l{100e-6}-r{47}c{1e-6}-r{47}");
	modelRange.clear();
	modelRange.push_back(eis::Range(1e1, 1e3, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(1e-6, 1e-3, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(1e1, 1e3, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(1e-9, 1e-5, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(1e1, 1e3, SWEEP_DEPTH/2, true));
	dataset->addModel(bord1_5, modelRange);

	eis::Model bord2("r{1e3}c{1e-6}-r{1e3}c{1e-9}-r{47}");
	modelRange.clear();
	modelRange.push_back(eis::Range(1e1, 1e4, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(1e-12, 1e-5, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(1e1, 1e4, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(1e-12, 1e-5, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(1e1, 1e4, SWEEP_DEPTH/2, true));
	dataset->addModel(bord2, modelRange);

	eis::Model bord4("r{100}-r{470}c{100e-12}-r{1e3}c{1e-9}-r{100}c{1e-6}");
	modelRange.clear();
	modelRange.push_back(eis::Range(10, 1e4, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(10, 1e4, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(1e-12, 1e-5, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(10, 1e4, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(1e-12, 1e-5, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(10, 1e4, SWEEP_DEPTH/2, true));
	modelRange.push_back(eis::Range(1e-12, 1e-5, SWEEP_DEPTH/2, true));
	dataset->addModel(bord4, modelRange);

	std::cout<<__func__<<" with "<<dataset->size().value()<<" examples\n";
	return dataset;
}

int main(int argc, char** argv)
{
	std::shared_ptr<Net> net(new Net);

	EisGeneratorDataset* trainDataset = getNeisGeneratedDataset();
	EisGeneratorDataset* testDataset = getNeisGeneratedDatasetAlt();

	torch::data::DataLoaderOptions options;
	options = options.batch_size(batch_size).workers(1);
	auto trainDataLoader = torch::data::make_data_loader(trainDataset->map(torch::data::transforms::Stack<>()), options);
	auto testDataLoader = torch::data::make_data_loader(testDataset->map(torch::data::transforms::Stack<>()), options);

	torch::optim::SGD optimizer(net->parameters(), torch::optim::SGDOptions(0.001).momentum(0.5));

	for (size_t i = 0; i < 2; ++i)
	{
		train(net, *trainDataLoader, optimizer, i+1, trainDataset->size().value());
		test(net, *testDataLoader, trainDataset->size().value());
	}

	delete trainDataset;
	delete testDataset;

	torch::save(net, "net.pt");

	return 0;
}
