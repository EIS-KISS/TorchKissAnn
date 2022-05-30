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
#include "trashgenerator.h"

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
	svmParam.gamma = 0.000977;	// 1/num_features if zero
	svmParam.coef0 = 0;
	svmParam.nu = 0.9;
	svmParam.cache_size = 10;
	svmParam.C = 4.1e3;
	svmParam.eps = 1e-3;
	svmParam.p = 0.1;
	svmParam.shrinking = 0;
	svmParam.probability = 1;
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
	EisGeneratorDataset* eisData = EisGeneratorDataset::getNeisGeneratedDatasetAlt(DATA_WIDTH);
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

template <typename DatasetSelf>
double testSvm(svm_model* model,
			   torch::data::datasets::Dataset<DatasetSelf, torch::data::Example<torch::Tensor, torch::Tensor>>* dataset,
			   bool oneClass = false)
{
	SvmLoader<DatasetSelf> svmloader(dataset);
	struct svm_problem problem = svmloader.getSvmProblem(SIZE_MAX, oneClass);
	int correct = 0;
	double acc;
	for(int i = 0; i < problem.l; ++i)
	{
		double prediction = svm_predict(model, problem.x[i]);
		if(prediction == problem.y[i] || (oneClass && prediction < 0))
			correct++;
		else
			Log(Log::DEBUG)<<"should be "<<(oneClass ? -1.0 : problem.y[i])<<" is "<<svm_predict(model, problem.x[i]);
	}

	acc = static_cast<double>(correct)/problem.l;

	if(oneClass)
	{
		int trashCorrect = 0;
		TrashDataset trashData;
		SvmLoader<TrashDataset> trashSvmloader(&trashData);
		struct svm_problem trashProblem = trashSvmloader.getSvmProblem(1000, oneClass);
		for(int i = 0; i < trashProblem.l; ++i)
		{
			double prediction = svm_predict(model, trashProblem.x[i]);
			if(prediction > 0)
				trashCorrect++;
			else
			{
				Log(Log::DEBUG)<<"trashSvmloader should be "<<1.0<<" is "<<svm_predict(model, problem.x[i]);
			}
		}
		double trashAcc = static_cast<double>(trashCorrect)/trashProblem.l;

		Log(Log::INFO)<<"Svm trash accuracy: "<<trashAcc;
		Log(Log::INFO)<<"Svm train accuracy: "<<acc;
		acc = (acc + trashAcc)/2;
	}

	Log(Log::INFO)<<"Svm accuracy: "<<acc;
	SvmLoader<DatasetSelf>::distroySvmProblem(problem);
	return acc;
}

void trainSvm(bool oneClass = false)
{
	EisGeneratorDataset* eisData = EisGeneratorDataset::getNeisGeneratedDataset(DATA_WIDTH);

	SvmLoader<EisGeneratorDataset> svmloader(eisData);
	struct svm_problem problem = svmloader.getSvmProblem(1000, oneClass);

	svm_parameter svmParam = getSvmParameters();
	if(!oneClass)
	{
		svmParam.probability = true;
		svmParam.svm_type = C_SVC;
		setSvmWeights(&svmParam, eisData->modelWeights());
	}
	else
	{
		svmParam.svm_type = ONE_CLASS;
		svmParam.probability = false;
	}

	svm_model* model = svm_train(&problem, &svmParam);

	SvmLoader<EisGeneratorDataset>::distroySvmProblem(problem);

	Log(Log::DEBUG)<<"svm type: "<<svm_get_svm_type(model);

	EisGeneratorDataset* eisDataAlt = EisGeneratorDataset::getNeisGeneratedDatasetAlt(DATA_WIDTH);
	testSvm<EisGeneratorDataset>(model, eisDataAlt, oneClass);

	constexpr char fileName[] = "./svm.vos";
	constexpr char fileNameOneClass[] = "./svmFilter.vos";
	svm_save_model(oneClass ? fileNameOneClass : fileName, model);
	svm_free_and_destroy_model(&model);
	Log(Log::INFO)<<"Saved svm model as "<<(oneClass ? fileNameOneClass : fileName);
}

int main(int argc, char** argv)
{
	std::cout<<std::setprecision(3)<<std::fixed<<std::setw(3);
	Log::level = Log::INFO;
	if(argc < 2)
	{
		Log(Log::ERROR)<<"You must specify what to train: ann, svm or svmcross";
		return -1;
	}

	svm_set_print_string_function(&svmPrintFunction);

	for(int i = 1; i < argc; ++i)
	{
		if(std::string(argv[i]) == "-v")
			Log::level = Log::DEBUG;
		else if(std::string(argv[i]) == "ann")
			trainAnn();
		else if(std::string(argv[i]) == "svm")
			trainSvm(false);
		else if(std::string(argv[i]) == "svmfilter")
			trainSvm(true);
		else if(std::string(argv[i]) == "svmcross")
			svmCross();
		else
			Log(Log::ERROR)<<argv[i]<<" is not a supported trainable model";
	}

	return 0;
}
