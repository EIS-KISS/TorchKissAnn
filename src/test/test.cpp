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
#include <algorithm>
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
#include "data/loaders/eisspectradataset.h"

// Rebuild the canonical grid recorded in the network onto a test dataset so the test spectra are
// resampled onto the same grid the network was trained on. No-op when the network has no grid.
template <typename T>
static void applyCanonicalFromNet(const std::shared_ptr<ann::Net>& net, T& dataset)
{
	if(!net->hasCanonicalGrid())
		return;
	EisSpectraDataset* spectra = dynamic_cast<EisSpectraDataset*>(&dataset);
	if(spectra == nullptr)
		return;
	CanonicalGrid grid;
	grid.enabled = true;
	grid.fMin = net->getCanonicalFMin();
	grid.fMax = net->getCanonicalFMax();
	grid.points = (size_t)net->getCanonicalPoints();
	grid.log = net->getCanonicalLog();
	grid.augmentShift = false; // never shift at test time
	spectra->setCanonicalGrid(grid);
	spectra->setCanonicalTraining(false);
}

// Apply the per spectra |Z| scaling recorded in the network to a test dataset, so the spectra are
// prepared the way the training loader prepared them. No-op for a network trained on raw
// impedances. Like the grid this has to happen before the dataset is handed to a data loader,
// which copies it.
template <typename T>
static void applySpectraNormFromNet(const std::shared_ptr<ann::Net>& net, T& dataset)
{
	if(net->getSpectraNorm().empty())
		return;
	EisSpectraDataset* spectra = dynamic_cast<EisSpectraDataset*>(&dataset);
	if(spectra == nullptr)
	{
		Log(Log::WARN)<<"The network expects every spectra scaled by its "<<net->getSpectraNorm()<<" |Z|, but this dataset is not built from spectra, the results are meaningless";
		return;
	}
	spectra->setSpectraNorm(spectraNormFromString(net->getSpectraNorm()));
	Log(Log::INFO)<<"Scaling every spectra by its "<<net->getSpectraNorm()<<" |Z| as recorded in the network";
}

/// Column header for output index i. Results are reported in the network's index space, so the
/// network's own labels are the right names whenever it stores them; the dataset names are only a
/// fallback for networks trained without labels.
template <typename T>
static std::string labelName(const std::vector<std::string>& netLabels, T& dataset, int64_t index)
{
	if(index >= 0 && static_cast<size_t>(index) < netLabels.size())
		return netLabels[static_cast<size_t>(index)];
	return dataset.outputName(index);
}

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
		// the dataset may carry a label remap applied by test() above, so size the buffers by the
		// network output width rather than by the dataset's own (pre remap) class count
		torch::Tensor weights =  torch::ones({net->getOutputSize()}).to(*offload_device);
		losses[i] = ann::classification::test(net, *dataLoader, dataset->size().value(), net->getOutputSize(), weights, dataset->isMulticlass()).loss;
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

	applyCanonicalFromNet(net, dataset);
	applySpectraNormFromNet(net, dataset);

	Log(Log::INFO)<<"Testing with"<<(dataset.isMulticlass() ? " muliclass" : "")<<
		" dataset of size "<<dataset.size().value()<<" with an output size of "<<
		dataset.outputSize()<<" and an input size of "<<dataset.inputSize();

	std::string type = tokenize(net->getPurpose(), ',')[0];
	if(type != "Classifier")
	{
		Log(Log::ERROR)<<"The network loaded from "<<config.netpath<<" is not a classifier network, can not test this network on this dataset!";
		return 1;
	}

	// The dataset interns classes in the order it happens to encounter them, which is the
	// directory order for EisDirDataset and the tar order for EisTarDataset, so the same classes
	// end up numbered differently depending on where they were read from. Accuracy, loss and the
	// confusion matrix all compare raw network output indices against dataset target indices, so
	// against a reinterned dataset the network scores on a permuted label space. Remap the targets
	// onto the network's own label order whenever the names line up; a dataset carrying classes the
	// network has never seen is a real mismatch and stays an error.
	// Note that this has to happen before the dataset is handed to the data loader: map() copies
	// the dataset by value, so a later mutation of it would be silently dropped.
	std::vector<std::string> netOutputs = net->getOutputLabels();
	int64_t evalOutputSize = net->getOutputSize();

	if(net->getOutputSize() < static_cast<int64_t>(dataset.outputSize()))
	{
		Log(Log::ERROR)<<"Network has "<<net->getOutputSize()<<" outputs but the dataset has "<<dataset.outputSize()<<" classes";
		return 1;
	}

	if(netOutputs.empty())
		Log(Log::WARN)<<"Network stores no output labels, can not verify or remap the dataset class order";
	else if(dataset.labelsInSameOrderAs(netOutputs))
	{
		// already numbered the way the network expects, leave the free identity path alone
	}
	else if(dataset.isMulticlass())
	{
		Log(Log::ERROR)<<"The dataset interns its classes in a different order than the network was trained with, and multiclass targets are multi hot vectors that can not be remapped by index. Reorder the dataset to match the network or train a new network on this dataset";
		return 1;
	}
	else if(!dataset.createLabelMap(*net))
	{
		for(size_t i = 0; i < dataset.outputSize(); ++i)
			if(std::find(netOutputs.begin(), netOutputs.end(), dataset.outputName(i)) == netOutputs.end())
				Log(Log::ERROR)<<"The dataset has the class "<<dataset.outputName(i)<<" which the network does not know about";
		Log(Log::ERROR)<<"Dataset and network do not describe the same set of classes, can not remap the dataset onto the network label order";
		return 1;
	}
	else
		Log(Log::INFO)<<"Dataset interns its classes in a different order than the network was trained with, remapped its targets onto the network label order";

	torch::data::DataLoaderOptions options;
	options = options.batch_size(batch_size).workers(16);
	auto dataLoader = torch::data::make_data_loader(dataset.map(torch::data::transforms::Stack<>()), options);

	// the weights are computed on the cpu, the loss has to see them on the device the network runs on
	ann::classification::TestReturn testRet = ann::classification::test(net, *dataLoader, dataset.size().value(), evalOutputSize,
		dataset.mapPerClassToNetOrder(dataset.classWeights(), evalOutputSize).to(*offload_device), dataset.isMulticlass());

	Log(Log::INFO)<<"Test loss: "<<testRet.loss;
	Log(Log::INFO)<<"Global Acc: "<<testRet.accOverall;
	Log(Log::INFO)<<"Class Acc:";

	for(int64_t i = 0; i < evalOutputSize; ++i)
		Log(Log::INFO, false)<<labelName(netOutputs, dataset, i)<<", ";
	Log(Log::INFO)<<'\n'<<tensorToString(testRet.acc);

	if(config.confustionMatrix)
	{
		torch::Tensor confustionMatrix = ann::classification::confusion(net, *dataLoader);
		Log(Log::INFO)<<"Confusion matrix:";
		for(int64_t i = 0; i < evalOutputSize; ++i)
			Log(Log::INFO, false)<<labelName(netOutputs, dataset, i)<<", ";
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

	applyCanonicalFromNet(net, dataset);
	applySpectraNormFromNet(net, dataset);

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

	Log(Log::INFO)<<"\nMse:\n"<<tensorToString(ret.mse)<<"\n\nR2:\n"<<tensorToString(ret.r2);

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
