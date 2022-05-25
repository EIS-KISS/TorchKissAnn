#include <cstddef>
#include <ios>
#include <iostream>
#include <eisgenerator/model.h>
#include <memory>
#include <torch/optim.h>
#include <libsvm/svm.h>

#include "net.h"
#include "eisgendata.h"
#include "testgenerator.h"
#include "log.h"
#include "svmloader.h"

#include "constants.h"

template <typename DataLoader>
void train(std::shared_ptr<Net> network, DataLoader& loader, torch::optim::Optimizer& optimizer,
		   size_t epoch, size_t data_size, torch::Tensor classWeights)
{
	std::cout<<__func__<<'\n';
	size_t index = 0;
	network->train();
	float Loss = 0, Acc = 0;

	for (auto& batch : loader)
	{
		auto data = batch.data.to(DATA_LOCATION);
		auto targets = batch.target.to(DATA_LOCATION).view({-1});

		auto prediction = network->forward(data);
		auto loss = torch::nll_loss(prediction, targets, classWeights);
		assert(!std::isnan(loss.template item<float>()));
		auto acc = prediction.argmax(1).eq(targets).sum();

		optimizer.zero_grad();
		loss.backward();
		optimizer.step();

		Loss += loss.template item<float>();
		Acc += acc.template item<float>();

		if (index++ % 1000 == 0)
		{
			auto end = std::min(data_size, (index + 1) * BATCH_SIZE);
			std::cout<<"Train Epoch: "<< epoch <<' '<<end<<'/'<< data_size<<"\tLoss: "<<Loss/end<<"\tAcc: "<<Acc/end<<'\n';
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
		auto data = batch.data.to(DATA_LOCATION);
		auto targets = batch.target.to(DATA_LOCATION).view({-1});

		auto output = network->forward(data);
		auto loss = torch::nll_loss(output, targets);
		assert(!std::isnan(loss.template item<float>()));
		auto acc = output.argmax(1).eq(targets).sum();

		Loss += loss.template item<float>();
		Acc += acc.template item<float>();
	}

	std::cout<<"Test Loss: "<<Loss / data_size<< "\tAcc: " <<Acc / data_size<< std::endl;
}

void trainAnn()
{
	EisGeneratorDataset* trainDataset = EisGeneratorDataset::getNeisGeneratedDatasetWithBLl(DATA_WIDTH);
	EisGeneratorDataset* testDataset = EisGeneratorDataset::getNeisGeneratedDatasetAlt(DATA_WIDTH);

	std::shared_ptr<Net> net(new Net(DATA_WIDTH, trainDataset->modelCount()));

	torch::data::DataLoaderOptions options;
	options = options.batch_size(BATCH_SIZE).workers(16);
	auto trainDataLoader = torch::data::make_data_loader(trainDataset->map(torch::data::transforms::Stack<>()), options);
	auto testDataLoader = torch::data::make_data_loader(testDataset->map(torch::data::transforms::Stack<>()), options);

	torch::optim::SGD optimizer(net->parameters(), torch::optim::SGDOptions(0.001).momentum(0.5));

	for (size_t i = 0; i < 10; ++i)
	{
		train(net, *trainDataLoader, optimizer, i+1, trainDataset->size().value(), trainDataset->modelWeights());
		test(net, *testDataLoader, testDataset->size().value());
		torch::save(net, "netUf.vos");
	}

	delete trainDataset;
	delete testDataset;

	torch::save(net, "net.vos");
}

svm_parameter getSvmParameters()
{
	svm_parameter svmParam;
	svmParam.svm_type = C_SVC;
	svmParam.kernel_type = RBF;
	svmParam.degree = 3;
	svmParam.gamma = 0.5;	// 1/num_features if zero
	svmParam.coef0 = 0;
	svmParam.nu = 0.5;
	svmParam.cache_size = 10;
	svmParam.C = 32;
	svmParam.eps = 1e-3;
	svmParam.p = 0.1;
	svmParam.shrinking = 1;
	svmParam.probability = 0;
	svmParam.nr_weight = 0;
	svmParam.weight_label = NULL;
	svmParam.weight = NULL;
	return svmParam;
}

void discardStr(const char* str)
{
	(void)str;
}

void svmCross()
{
	svm_set_print_string_function(&discardStr);
	EisGeneratorDataset* eisData = EisGeneratorDataset::getNeisGeneratedDataset(DATA_WIDTH);
	SvmLoader<EisGeneratorDataset> svmloader(eisData);
	svm_problem problem = svmloader.getSvmProblem(2000);
	svm_parameter params = getSvmParameters();

	double bestAcc = 0;
	double bestGamma = 0;
	double bestC = 0;

	std::cout<<std::setprecision(3)<<std::defaultfloat<<std::setw(3);
	size_t iteration = 0;

	constexpr int gammaStart = -15;
	constexpr int gammaEnd = 3;
	constexpr int cStart = -5;
	constexpr int cEnd = 15;

	for(int gamma = gammaStart; gamma < gammaEnd+1; ++gamma)
	{
		for(int c = cStart; c < cEnd+1; ++c)
		{
			double* target = new double[problem.l];
			params.C = pow(2, c);
			params.gamma = pow(2, gamma);

			svm_cross_validation(&problem, &params, 10, target);

			size_t correctCount = 0;
			for(int i = 0; i < problem.l; ++i)
			{
				if(target[i] == problem.y[i])
					++correctCount;
			}

			double acc = static_cast<double>(correctCount)/problem.l;
			if(acc > bestAcc)
			{
				bestGamma = params.gamma;
				bestC = params.C;
				bestAcc = acc;
			}

			Log(Log::INFO)<<"Iteration "<<iteration++<<" of "<<(gammaEnd-gammaStart)*(cEnd-cStart);
			Log(Log::INFO)<<"ACC: "<<static_cast<double>(correctCount)/problem.l<<" Gamma: "<<params.gamma<<" C: "<<params.C;
		}
	}

	Log(Log::INFO)<<"Reccomended param values:\nGamma: "<<bestGamma<<" C: "<<bestC<<" ACC: "<<bestAcc;
}

void trainSvm()
{
	EisGeneratorDataset* eisData = EisGeneratorDataset::getNeisGeneratedDataset(DATA_WIDTH);

	SvmLoader<EisGeneratorDataset> svmloader(eisData);
	struct svm_problem problem = svmloader.getSvmProblem(100000);

	svm_parameter svmParam = getSvmParameters();

	svm_model* model = svm_train(&problem, &svmParam);

	SvmLoader<EisGeneratorDataset>::distroySvmProblem(problem);

	constexpr char fileName[] = "./svm.vos";
	svm_save_model(fileName, model);
	svm_free_and_destroy_model(&model);
	Log(Log::INFO)<<"Saved svm model as "<<fileName;
}

int main(int argc, char** argv)
{
	std::cout<<std::setprecision(3)<<std::fixed<<std::setw(3);
	Log::level = Log::DEBUG;
	if(argc < 2)
	{
		Log(Log::ERROR)<<"You must specify what to train: ann, svm or svmcross";
		return -1;
	}

	if(std::string(argv[1]) == "ann")
		trainAnn();
	else if(std::string(argv[1]) == "svm")
		trainSvm();
	else if(std::string(argv[1]) == "svmcross")
		svmCross();
	else
		Log(Log::ERROR)<<argv[1]<<" is not a support trainable model";

	return 0;
}
