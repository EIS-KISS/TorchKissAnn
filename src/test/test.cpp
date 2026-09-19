//
// TorchKissAnn - A collection of tools to train various types of Machine learning
// algorithms on various types of EIS data
// Copyright (C) 2025 Carl Klemm <carl@uvos.xyz>
//
// This file is part of TorchKissAnn.
//
// TorchKissAnn is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// TorchKissAnn is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with TorchKissAnn.  If not, see <http://www.gnu.org/licenses/>.
//

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <eisgenerator/model.h>
#include <eisgenerator/log.h>
#include <memory>
#include <torch/torch.h>
#include <valarray>

#include "ann/autoencoder.h"
#include "ann/autoencode.h"
#include "ann/classification.h"
#include "ann/regression.h"
#include "commonoptions.h"
#include "data/eisdataset.h"
#include "data/loaders/tarloader.h"
#include "log.h"
#include "data/loaders/dirloader.h"
#include "data/loaders/regressiondirloader.h"
#include "data/loaders/regressionloader.h"
#include "options.h"
#include "globals.h"
#include "tensoroperators.h"
#include "tensoroptions.h"
#include "tokenize.h"
#include "ploting.h"
#include "save.h"
#include "colortabel.h"
#include "data/eistotorch.h"

template <typename T>
int inputImportance(std::shared_ptr<ann::Net> net, T* dataset, int64_t window, const Config& config)
{
	std::vector<double> losses(dataset->inputSize());

	std::pair<torch::Tensor, torch::Tensor> ranges = dataset->inputRanges();

	for(size_t i = 0; i < dataset->inputSize(); ++i)
	{
		std::vector<DropDesc> desc(dataset->inputSize(), {0, 0, 0, 0});
		for(size_t j = (static_cast<int64_t>(i) - window) > 0 ? static_cast<int64_t>(i) - window : 0; j <= i+window && j < dataset->inputSize(); ++j)
		{
			desc[j].dropout = true;
			desc[j].max = ranges.first[j].item<float>();
			desc[j].min = ranges.second[j].item<float>();
			desc[j].strength = 1;
		}

		dataset->setDropouts(desc);

		torch::data::DataLoaderOptions options;
		options = options.batch_size(batch_size).workers(16);
		auto dataLoader = torch::data::make_data_loader(dataset->map(torch::data::transforms::Stack<>()), options);
		torch::Tensor weights =  torch::ones({static_cast<int64_t>(dataset->outputSize())}).to(*offload_device);
		losses[i] = ann::classification::test(net, *dataLoader, dataset->size().value(), dataset->outputSize(), weights, dataset->isMulticlass()).loss;
	}

	std::vector<double> indecies(dataset->inputSize());
	for(size_t i = 0; i < dataset->inputSize(); ++i)
		indecies[i] = i;

	csv::save(config.outputDir/("loss_importance"+std::to_string(window)+".svg"), losses, "loss importance");
	save2dPlot(config.outputDir/("loss_importance"+std::to_string(window)+".svg"), "Input", "loss", indecies, losses, false, false, true);

	return 0;
}

template <typename T>
int test(std::shared_ptr<ann::Net> net, const Config& config)
{
	if(config.fileName.empty())
	{
		Log(Log::ERROR)<<"Please specify a filename of a valid dataset file";
		return 1;
	}

	T dataset(config.fileName);

	if(dataset.size().value() == 0)
	{
		Log(Log::ERROR)<<"Failed to create dataset for "<<config.fileName;
		return 2;
	}

	Log(Log::INFO)<<"Testing with"<<(dataset.isMulticlass() ? " muliclass" : "")<<
		" dataset of size "<<dataset.size().value()<<" with an output size of "<<
		dataset.outputSize()<<" and an input size of "<<dataset.inputSize();

	std::string type = tokenize(net->getPurpose(), ',')[0];
	if(type != "Classifier")
	{
		Log(Log::ERROR)<<"The network loaded from "<<config.netpath<<" is not a classifier network, can not test this network on this dataset!";
		return 1;
	}

	torch::data::DataLoaderOptions options;
	options = options.batch_size(batch_size).workers(16);
	auto dataLoader = torch::data::make_data_loader(dataset.map(torch::data::transforms::Stack<>()), options);

	ann::classification::TestReturn testRet = ann::classification::test(net, *dataLoader, dataset.size().value(), dataset.outputSize(), dataset.classWeights(), dataset.isMulticlass());

	Log(Log::INFO)<<"Test loss:"<<testRet.loss;
	Log(Log::INFO)<<"Acc:";

	for(size_t i = 0; i < dataset.outputSize(); ++i)
		Log(Log::INFO, false)<<dataset.outputName(i)<<", ";
	Log(Log::INFO)<<'\n'<<tensorToString(testRet.acc);

	if(config.confustionMatrix)
	{
		torch::Tensor confustionMatrix = ann::classification::confusion(net, *dataLoader);
		Log(Log::INFO)<<"Confusion matrix:";
		for(size_t i = 0; i < dataset.outputSize(); ++i)
			Log(Log::INFO, false)<<dataset.outputName(i)<<", ";
		Log(Log::INFO)<<'\n'<<tensorToString(confustionMatrix)<<'\n';
	}

	if(config.inputImportance)
	{
		inputImportance<T>(net, &dataset, 1, config);
		inputImportance<T>(net, &dataset, 4, config);
		inputImportance<T>(net, &dataset, 16, config);
	}

	return 0;
}

template <typename T>
int inputImportanceRegression(std::shared_ptr<ann::Net> net, T* dataset, int64_t window, const Config& config)
{
	std::vector<ann::regression::TestReturn> returns(dataset->inputSize());

	std::pair<torch::Tensor, torch::Tensor> ranges = dataset->inputRanges();

	for(size_t i = 0; i < dataset->inputSize(); ++i)
	{
		std::vector<DropDesc> desc(dataset->inputSize(), {0, 0, 0, 0});
		for(size_t j = (static_cast<int64_t>(i) - window) > 0 ? static_cast<int64_t>(i) - window : 0; j <= i+window && j < dataset->inputSize(); ++j)
		{
			desc[j].dropout = true;
			desc[j].max = ranges.first[j].item<float>();
			desc[j].min = ranges.second[j].item<float>();
			desc[j].strength = 1;
		}

		dataset->setDropouts(desc);

		torch::data::DataLoaderOptions options;
		options = options.batch_size(batch_size).workers(16);
		auto dataLoader = torch::data::make_data_loader(dataset->map(torch::data::transforms::Stack<>()), options);
		std::unique_ptr<torch::nn::MSELoss> lossMse(new torch::nn::MSELoss(torch::nn::MSELossOptions().reduction(torch::kMean)));
		returns[i] = ann::regression::test(net, *dataLoader, *lossMse, dataset->size().value());
	}

	std::valarray<double> indecies(dataset->inputSize());
	std::vector<torch::Tensor> r2s(dataset->inputSize());
	std::valarray<double> loss(dataset->inputSize());
	for(size_t i = 0; i < dataset->inputSize(); ++i)
	{
		indecies[i] = i;
		ann::regression::TestReturn ret = returns[i];
		r2s[i] = ret.r2;
		loss[i] = ret.loss;
	}

	for(size_t i = 0; i < dataset->outputSize(); ++i)
	{
		std::valarray<double> r2(dataset->inputSize());
		for(size_t j = 0; j < dataset->inputSize(); ++j)
			r2[j] = std::max(r2s[j][i].item<float>(), 0.0f);
		csv::save(config.outputDir/("r2_"+std::to_string(i)+"_importance_"+std::to_string(window)+".csv"), r2, "r2");
		save2dPlot(config.outputDir/("r2_"+std::to_string(i)+"_importance_"+std::to_string(window)+".svg"), "Input", "r2", indecies, r2, false, false, true);
	}
	csv::save(config.outputDir/("loss_importance"+std::to_string(window)+".svg"), loss, "loss importance");
	save2dPlot(config.outputDir/("loss_importance"+std::to_string(window)+".svg"), "Input", "loss", indecies, loss, false, false, true);

	return 0;
}

template <typename T>
int testRegression(std::shared_ptr<ann::Net> net, const Config& config)
{
	if(config.fileName.empty())
	{
		Log(Log::ERROR)<<"Please specify a filename of a valid dataset file";
		return 1;
	}

	T dataset(config.fileName);

	if(dataset.size().value() == 0)
	{
		Log(Log::ERROR)<<"Failed to create dataset for "<<config.fileName;
		return 2;
	}

	Log(Log::INFO)<<"Testing with"<<(dataset.isMulticlass() ? " muliclass" : "")<<
		" dataset of size "<<dataset.size().value()<<" with an output size of "<<
		dataset.outputSize()<<" and an input size of "<<dataset.inputSize();

	std::string type = tokenize(net->getPurpose(), ',')[0];
	if(type != "Regression")
	{
		Log(Log::ERROR)<<"The network loaded from "<<config.netpath<<" is not a classifier network, can not test this network on this dataset!";
		return 1;
	}

	Log(Log::INFO)<<"Network was trained to be "<<net->getPurpose()<<", we are testing against "<<ann::regression::purposeString(&dataset);

	if(net->getOutputSize() !=  static_cast<int64_t>(dataset.outputSize()))
	{
		Log(Log::ERROR)<<"Network has "<<net->getOutputSize()<<" outputs but the dataset has "<<dataset.outputSize()<<" outputs";
		return 1;
	}
	else
	{
		std::vector<std::string> netOutputs = net->getOutputLabels();
		for(size_t i = 0; i < netOutputs.size(); ++i)
		{
			if(netOutputs[i] != dataset.outputName(i))
			{
				Log(config.ignoreMissmatch ? Log::INFO : Log::ERROR)<<"Network has "<<netOutputs[i]<<" as its "<<i<<(i != 1 ? "st" : "th")<<" output but the dataset has "<<dataset.outputName(i);
				if(!config.ignoreMissmatch)
					return 1;
			}
		}
	}

	if(net->getExtraInputs().size() != dataset.extraInputs().size())
	{
		Log(Log::ERROR)<<"Network requries "<<net->getExtraInputs().size()<<" extra inputs but the dataset has "<<dataset.extraInputs().size()<<" extra inputs";
		return 1;
	}
	else
	{
		std::vector<std::pair<std::string, int64_t>> netExtraInputs = net->getExtraInputs();
		std::vector<std::pair<std::string, int64_t>> dataExtraInputs = dataset.extraInputs();
		for(size_t i = 0; i < net->getExtraInputs().size(); ++i)
		{
			if(netExtraInputs[i].first != dataExtraInputs[i].first)
			{
				Log(Log::ERROR)<<"Network requries "<<netExtraInputs[i].first<<" as extra input "<<i<<" but the dataset dose not provide this";
				return 1;
			}
		}
	}

	torch::data::DataLoaderOptions options;
	options = options.batch_size(batch_size).workers(16);
	auto dataLoader = torch::data::make_data_loader(dataset.map(torch::data::transforms::Stack<>()), options);

	std::unique_ptr<torch::nn::MSELoss> lossMse(new torch::nn::MSELoss(torch::nn::MSELossOptions().reduction(torch::kMean)));
	ann::regression::TestReturn ret = ann::regression::test(net, *dataLoader, *lossMse, dataset.size().value());

	torch::Tensor targets = ret.targets.contiguous();
	auto targetAccessor = targets.accessor<float, 2>();
	torch::Tensor predictions = ret.predictions.contiguous();
	auto predictionsAccessor = predictions.accessor<float, 2>();

	std::filesystem::create_directory(config.outputDir);

	for(int64_t output = 0; output < net->getOutputSize(); ++output)
	{
		std::valarray<double> targetsVarr(targets.size(0));
		std::valarray<double> predictionsVarr(targets.size(0));
		std::valarray<double> sequence(targets.size(0));
		std::valarray<double> error(targets.size(0));
		for(int64_t i = 0; i < targets.size(0); ++i)
		{
			targetsVarr[i] = targetAccessor[i][output];
			predictionsVarr[i] = predictionsAccessor[i][output];

			sequence[i] = i;
			error[i] = (predictionsVarr[i] - targetsVarr[i])/targetsVarr[i];
		}
		csv::save(config.outputDir/(net->getOutputLabels()[output]+"_target.csv"), targetsVarr, "target");
		csv::save(config.outputDir/(net->getOutputLabels()[output]+"_prediction.csv"), predictionsVarr, "prediction");
		save2dPlot(config.outputDir/(net->getOutputLabels()[output]+".svg"), "Target", "Prediction", targetsVarr, predictionsVarr, false, false, true);
		save2dPlot(config.outputDir/(net->getOutputLabels()[output]+"_error.svg"), net->getOutputLabels()[output], "Error", targetsVarr, error, false, false, true);
	}

	Log(Log::INFO)<<"\nMse:\n"<<tensorToString(ret.mse)<<"\n\n R2:\n"<<tensorToString(ret.r2);

	if(config.inputImportance)
	{
		inputImportanceRegression<T>(net, &dataset, 1, config);
		inputImportanceRegression<T>(net, &dataset, 4, config);
		inputImportanceRegression<T>(net, &dataset, 16, config);
	}

	return 0;
}

template <typename T>
int testAutoencoder(std::shared_ptr<ann::AutoEncoder> net, const Config& config)
{
	T dataset(config.fileName);

	if(dataset.size().value() == 0)
	{
		Log(Log::ERROR)<<"Failed to create dataset for "<<config.fileName;
		return 2;
	}

	std::string type = tokenize(net->getPurpose(), ',')[0];
	if(type != "Autoencoder")
	{
		Log(Log::ERROR)<<"The network loaded from "<<config.netpath<<" is not a autoencoder network, can not test this network on this dataset!";
		return 1;
	}

	Log(Log::INFO)<<"Network was trained to be "<<net->getPurpose()<<'\n';

	torch::data::DataLoaderOptions options;
	options = options.batch_size(batch_size).workers(16);
	auto dataLoader = torch::data::make_data_loader(dataset.map(torch::data::transforms::Stack<>()), options);
	std::unique_ptr<torch::nn::MSELoss> lossMse(new torch::nn::MSELoss(torch::nn::MSELossOptions().reduction(torch::kMean)));
	ann::autoencode::TestReturn ret = ann::autoencode::test(net, *dataLoader,  *lossMse, dataset.size().value());

	Log(Log::INFO)<<"Mse: "<<ret.mse.mean().item()<<'\n';

	assert(ret.latents.dim() == 2);

	auto latentAccesor = ret.latents.accessor<float, 2>();
	std::valarray<double> x(ret.latents.size(0));
	std::valarray<double> y(ret.latents.size(0));
	std::valarray<double> z(ret.latents.size(0));

	std::filesystem::create_directory(config.outputDir);

	std::cout<<"target sizes: \n"<<ret.datasetTargets.sizes()<<std::endl;

	for(int64_t i = 0; i < ret.latents.size(0); ++i)
	{
		switch(ret.latents.size(1))
		{
			default:
			case 3:
				z[i] = latentAccesor[i][2];
			case 2:
				y[i] = latentAccesor[i][1];
			case 1:
				x[i] = latentAccesor[i][0];
				break;
			case 0:
				assert(false);
		}
	}

	csv::save(config.outputDir/"latents.csv", ret.latents, "x ,y, z");

	if(config.outputMoreImages)
	{
		std::filesystem::path imgDir = config.outputDir/"images";
		std::filesystem::create_directory(imgDir);
		for(int64_t i = 0; i < ret.outputs.size(0); ++i)
		{
			std::pair<std::valarray<double>, std::valarray<double>> output = torchToValarray<double>(ret.outputs[i].to(torch::kFloat64));
			std::pair<std::valarray<double>, std::valarray<double>> referance = torchToValarray<double>(ret.inputs[i].to(torch::kFloat64));

			save2dPlot(imgDir/("output_" + std::to_string(i) + ".png"), "Re(Z)", "Im(Z)", {output, referance});
		}
	}

	if(ret.latents.size(0) == ret.datasetTargets.size(0) && ret.datasetTargets.size(1) == 1)
	{
		torch::Tensor metadata = torch::zeros({ret.latents.size(0), 4}, tensorOptCpu<float>(false));
		auto metadataAcesssor = metadata.accessor<float, 2>();
		for(int64_t i = 0; i < ret.latents.size(0); ++i)
		{
			std::array<int, 3> rgb = color::getColor(static_cast<color::color>(ret.datasetTargets[i].item().toInt()+1));
			metadataAcesssor[i][0] = ret.datasetTargets[i].item().toFloat();
			metadataAcesssor[i][1] = rgb[0];
			metadataAcesssor[i][2] = rgb[1];
			metadataAcesssor[i][3] = rgb[2];
		}

		csv::save(config.outputDir/"latents-with-meta.csv", torch::cat({ret.latents, metadata}, 1), "x ,y, z, class, r, g, b");
	}

	switch(ret.latents.size(1))
	{
		default:
		case 3:
			save3dPlot(config.outputDir/"latent.svg", "x", "y", "z", x, y, z, false);
			break;
		case 2:
			save2dPlot(config.outputDir/"latent.svg", "x", "y", x, y, false, false, true);
			break;
		case 1:
			for(int64_t i = 0; i < ret.latents.size(0); ++i)
				y[i] = i;
			save2dPlot(config.outputDir/"latent.svg", "n", "x", y, x, false, false, true);
			break;
		case 0:
			assert(false);
	}

	return 0;
}

int main(int argc, char** argv)
{
	std::cout<<std::setprecision(5)<<std::fixed<<std::setw(3);
	Log::level = Log::INFO;
	eis::Log::level = eis::Log::ERROR;

	Config config;
	argp_parse(&argp, argc, argv, 0, 0, &config);

	choose_device(config.noGpu);
	batch_size = config.batchSize;

	if(config.netpath.empty())
	{
		Log(Log::ERROR)<<"A path to a checkpoint directory must be given";
		return 1;
	}

	std::shared_ptr<ann::Net> net(ann::Net::newNetFromCheckpointDir(config.netpath));

	if(!net)
	{
		Log(Log::ERROR)<<"Could not load network from "<<config.netpath;
		return 1;
	}
	std::shared_ptr<ann::AutoEncoder> autoencoder = std::dynamic_pointer_cast<ann::AutoEncoder>(net);


	if(autoencoder)
	{
		switch(config.datasetMode)
		{
			case DATASET_DIR:
				return testAutoencoder<EisDirDataset>(autoencoder, config);
			case DATASET_TAR:
				return testAutoencoder<EisTarDataset>(autoencoder, config);
			case DATASET_DIR_REGRESSION:
				return testAutoencoder<RegressionLoaderDir>(autoencoder, config);
			case DATASET_TAR_REGRESSION:
				return testAutoencoder<RegressionLoaderTar>(autoencoder, config);
			case DATASET_INVALID:
				Log(Log::ERROR)<<"You must specify a valid dataset to use: " DATASET_LIST;
				break;
		}
	}
	else
	{
		switch(config.datasetMode)
		{
			case DATASET_DIR:
				return test<EisDirDataset>(net, config);
			case DATASET_TAR:
				return test<EisTarDataset>(net, config);
			case DATASET_DIR_REGRESSION:
				return testRegression<RegressionLoaderDir>(net, config);
			case DATASET_TAR_REGRESSION:
				return testRegression<RegressionLoaderTar>(net, config);
			case DATASET_INVALID:
				Log(Log::ERROR)<<"You must specify a valid dataset to use: " DATASET_LIST;
				break;
		}
	}

	free_device();
	return 0;
}
