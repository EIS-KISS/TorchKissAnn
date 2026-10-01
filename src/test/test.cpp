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
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <eisgenerator/model.h>
#include <eisgenerator/log.h>
#include <map>
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
#include "modelscript.h"
#include "ploting.h"
#include "save.h"
#include "colortabel.h"
#include "data/eistotorch.h"
#include "data/loaders/eisspectradataset.h"
#include "fit/fit.h"
#include "loss/circuitsimplicity.h"
#include "loss/simplicitytolerantloss.h"

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

// Turn on the set valued labels of -s simplicity. The band over which an element has to stay out of
// the impedance is the one the network is shown, so this needs the dataset after its canonical grid
// has been rebuilt from the network, and it has to happen before the data loader copies the dataset.
// Returns false, leaving the plain labels in place, when the dataset cannot describe them.
template <typename T>
static bool enableSimplicityLabels(T& dataset, const Config& config)
{
	if(dataset.isMulticlass())
	{
		Log(Log::WARN)<<"-s simplicity needs one circuit per class, a multiclass dataset can not carry a set of acceptable circuits, scoring with the plain labels";
		return false;
	}

	std::vector<double> omegas = circuit::uniqueOmegas(dataset.frequencies().value_or(torch::Tensor()));
	if(omegas.empty())
	{
		Log(Log::WARN)<<"Dataset does not report a frequency axis, testing element contributions on a fallback grid from 0.01 to 100000 rad/s";
		omegas.resize(32);
		for(size_t i = 0; i < omegas.size(); ++i)
			omegas[i] = std::pow(10.0, -2.0 + 6.0*static_cast<double>(i)/static_cast<double>(omegas.size()-1));
	}

	dataset.enableSimplicityLabels(omegas, config.simplicityPreference, config.simplicityThreshold);
	if(not dataset.hasSimplicityLabels())
	{
		Log(Log::WARN)<<"-s simplicity needs the generating circuit of every spectra, which this dataset does not carry, scoring with the plain labels";
		return false;
	}
	return true;
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

/// One misclassified spectra together with the circuit the network named for it.
struct Mismatch
{
	int64_t index;
	std::string actual;
	std::string predicted;
	torch::Tensor values;
	torch::Tensor omegas;
};

const char* fitVerdictName(const FitResult& fit)
{
	if(not fit.success)
		return "no fit";
	switch(fit.verdict)
	{
		case FitVerdict::FIT_VERDICT_FIT:
			return "fits";
		case FitVerdict::FIT_VERDICT_MARGINAL:
			return "marginal";
		case FitVerdict::FIT_VERDICT_UNFITTED:
			return "does not fit";
	}
	return "unknown";
}

/**
 * Fit the circuit the network predicted to every spectra it labelled wrongly.
 *
 * Naming the wrong class is only a real failure when no circuit describes the spectra at all: answer
 * "lr-p" for a spectra that is really "r-p" and the network has still handed the user a model they
 * can work with, one whose parameters mean something. eisFitBatch() decides that per spectra, and
 * the after fit accuracy counts a misclassification as good as right whenever the circuit the
 * network named fits. Marginal verdicts are reported but not counted, they say the circuit misses a
 * feature of the measurement.
 *
 * Mismatches are grouped by the circuit they were predicted to, because one batch fits one circuit,
 * and within that by their frequency grid, because one batch shares one grid.
 */
template <typename T>
void afterFitAccuracy(const std::shared_ptr<ann::Net>& net, T& dataset, const Config& config,
		const std::vector<std::string>& netOutputs, const torch::Tensor& predictions,
		const torch::Tensor& targets)
{
	if(netOutputs.empty())
	{
		Log(Log::WARN)<<"--fit-mismatch needs a network that stores the circuit of every class as its output labels, "<<config.netpath<<" stores none";
		return;
	}
	if(dataset.isMulticlass())
	{
		Log(Log::WARN)<<"--fit-mismatch needs one circuit per class, a multiclass network answers with a set of them, skipping";
		return;
	}

	torch::Tensor predicted = predictions.to(torch::kCPU).argmax(1).to(torch::kLong).reshape({-1});
	torch::Tensor actual = targets.to(torch::kCPU).to(torch::kLong).reshape({-1});
	const int64_t total = std::min(predicted.numel(), actual.numel());

	int64_t correct = 0;
	std::vector<Mismatch> mismatches;
	for(int64_t i = 0; i < total; ++i)
	{
		int64_t guess = predicted[i].item<int64_t>();
		int64_t truth = actual[i].item<int64_t>();
		if(guess == truth)
		{
			++correct;
			continue;
		}
		if(guess < 0 || static_cast<size_t>(guess) >= netOutputs.size())
		{
			Log(Log::WARN)<<"the network answered with output "<<guess<<" for spectra "<<i<<", which carries no label, can not fit that";
			continue;
		}

		std::vector<eis::DataPoint> measured = dataset.rawSpectraAtIndex(static_cast<size_t>(i));
		if(measured.size() < 3)
		{
			Log(Log::WARN)<<"spectra "<<i<<" has only "<<measured.size()<<" points, its mismatch is left out of the after fit accuracy";
			continue;
		}
		std::sort(measured.begin(), measured.end());
		torch::Tensor omegas;
		torch::Tensor values = eisToComplexTensor(measured, &omegas);
		mismatches.push_back({i, labelName(netOutputs, dataset, truth), netOutputs[static_cast<size_t>(guess)],
				values, omegas});
	}

	Log(Log::INFO)<<"\nFitting the predicted circuit to the "<<mismatches.size()<<" of "<<total
		<<" spectra the network labelled wrongly";

	// A plausibility check, not a parameter extraction: the fit only has to get near enough to the
	// minimum for the verdict to be right. The single fit's 1e-10 tolerance never stops on measured
	// data, whose cost keeps falling along flat directions, so every start would burn its whole
	// budget and a run over a real test set would take hours.
	FitOptions options;
	options.restarts = std::max<int64_t>(0, config.fitRestarts);
	options.maxEvaluations = std::max<int64_t>(1, config.fitEvaluations);
	options.fitLimit = config.fitLimit;
	options.searchPoints = config.fitPoints;
	options.tolerance = 1e-4;
	const int64_t batch = std::max<int64_t>(1, config.fitBatchSize);

	std::map<std::pair<std::string, std::vector<double>>, std::vector<size_t>> groups;
	for(size_t k = 0; k < mismatches.size(); ++k)
	{
		torch::Tensor grid = mismatches[k].omegas.to(torch::kFloat64);
		std::vector<double> key(grid.size(0));
		std::copy_n(grid.data_ptr<double>(), grid.size(0), key.begin());
		groups[{mismatches[k].predicted, std::move(key)}].push_back(k);
	}

	std::vector<std::string> verdict(mismatches.size(), "no fit");
	std::vector<double> rms(mismatches.size(), 0);
	// per predicted circuit: mismatches, fits, marginal, rejected
	std::map<std::string, std::array<int64_t, 4>> perCircuit;
	int64_t done = 0;

	for(const auto& [circuit, members]: groups)
	{
		std::array<int64_t, 4>& row = perCircuit[circuit.first];
		row[0] += static_cast<int64_t>(members.size());

		for(size_t begin = 0; begin < members.size(); begin += static_cast<size_t>(batch))
		{
			size_t end = std::min(members.size(), begin+static_cast<size_t>(batch));

			std::vector<torch::Tensor> rows;
			rows.reserve(end-begin);
			for(size_t k = begin; k < end; ++k)
				rows.push_back(mismatches[members[k]].values);

			std::vector<FitResult> results;
			try
			{
				results = eisFitBatch(torch::stack(rows), mismatches[members[begin]].omegas,
						circuit.first, options);
			}
			catch(const std::exception& error)
			{
				// a class label that is not a circuit eis::Model can compile: those mismatches stay
				// unfitted rather than taking the whole report down
				Log(Log::WARN)<<"the class "<<circuit.first<<" is not a circuit the fit can use: "<<error.what();
				continue;
			}

			for(size_t k = begin; k < end; ++k)
			{
				const FitResult& fit = results[k-begin];
				verdict[members[k]] = fitVerdictName(fit);
				rms[members[k]] = fit.rmsRelativeError;
				if(fit.success && fit.verdict == FitVerdict::FIT_VERDICT_FIT)
					row[1]++;
				else if(fit.success && fit.verdict == FitVerdict::FIT_VERDICT_MARGINAL)
					row[2]++;
				else
					row[3]++;
			}

			done += static_cast<int64_t>(end-begin);
			Log(Log::INFO)<<"  fitted "<<done<<" of "<<mismatches.size()<<" mismatches to "<<circuit.first;
		}
	}

	int64_t fits = 0, marginal = 0, rejected = 0;
	for(const auto& [circuit, row]: perCircuit)
	{
		fits += row[1];
		marginal += row[2];
		rejected += row[3];
	}

	const double accuracy = total > 0 ? static_cast<double>(correct+fits)/static_cast<double>(total) : 0;

	Log(Log::INFO)<<"\nAfter fit accuracy";
	Log(Log::INFO)<<"  classified correctly        "<<correct<<" of "<<total<<" ("<<correct/double(total)*100<<" %)";
	Log(Log::INFO)<<"  misclassified               "<<mismatches.size();
	Log(Log::INFO)<<"  predicted circuit fits      "<<fits;
	Log(Log::INFO)<<"  predicted circuit marginal  "<<marginal<<" (not counted)";
	Log(Log::INFO)<<"  predicted circuit rejected  "<<rejected;
	Log(Log::INFO)<<"  accuracy after the fit      "<<accuracy*100<<" %";

	Log(Log::INFO)<<"\n  Predicted circuit       Mismatches   Fits   Marginal   Rejected";
	for(const auto& [circuit, row]: perCircuit)
		Log(Log::INFO)<<'\n'<<std::left<<std::setw(25)<<circuit<<std::right<<std::setw(10)<<row[0]
			<<std::setw(8)<<row[1]<<std::setw(11)<<row[2]<<std::setw(10)<<row[3];
	Log(Log::INFO)<<'\n';

	std::filesystem::create_directories(config.outputDir);
	std::ofstream csv(config.outputDir/"afterfit.csv");
	if(not csv)
	{
		Log(Log::WARN)<<"could not write "<<config.outputDir/"afterfit.csv";
		return;
	}
	csv<<"index,target,predicted,verdict,rms_relative\n";
	for(size_t k = 0; k < mismatches.size(); ++k)
		csv<<mismatches[k].index<<",\""<<mismatches[k].actual<<"\",\""<<mismatches[k].predicted
			<<"\","<<verdict[k]<<","<<std::setprecision(6)<<rms[k]<<'\n';
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

	PlotOptions options;
	options.type = PLOT_TYPE_SCATTER;
	options.xLabel = "Input";
	options.yLabel = "Loss";

	save2dPlot(config.outputDir/("loss_importance"+std::to_string(window)+".svg"), indecies, losses, options);

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
	bool simplicity = (config.loss == LOSS_SIMPLICITY) and enableSimplicityLabels(dataset, config);

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
	// sequential, the per spectra diagnostics below index the dataset by the position a sample
	// came out of the loader, which a random sampler would shuffle
	auto dataLoader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
			dataset.map(torch::data::transforms::Stack<>()), options);

	// the weights are computed on the cpu, the loss has to see them on the device the network runs on
	torch::Tensor evalWeights = dataset.mapPerClassToNetOrder(dataset.classWeights(), evalOutputSize).to(*offload_device);
	std::unique_ptr<SimplicityTolerantLoss> simplicityLoss;
	if(simplicity)
		simplicityLoss.reset(new SimplicityTolerantLoss(evalWeights));

	ann::classification::TestReturn testRet = ann::classification::test(net, *dataLoader, dataset.size().value(), evalOutputSize,
		evalWeights, dataset.isMulticlass(), 0, nullptr, simplicityLoss.get());

	Log(Log::INFO)<<"Test loss: "<<testRet.loss;
	Log(Log::INFO)<<"Global Acc: "<<testRet.accOverall;
	if(simplicity)
		Log(Log::INFO)<<"Acceptable Acc: "<<testRet.simplicityAccOverall<<" (predicted circuit is one the spectra can not rule out)";
	Log(Log::INFO)<<"Class Acc:";

	for(int64_t i = 0; i < evalOutputSize; ++i)
		Log(Log::INFO, false)<<labelName(netOutputs, dataset, i)<<", ";
	Log(Log::INFO)<<'\n'<<tensorToString(testRet.acc, false);

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

	if(config.fitMismatch)
		afterFitAccuracy<T>(net, dataset, config, netOutputs, testRet.predictions, testRet.targets);

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

	PlotOptions options;
	options.type = PLOT_TYPE_SCATTER;
	options.xLabel = "Input";
	options.yLabel = "r2";

	for(size_t i = 0; i < dataset->outputSize(); ++i)
	{
		std::valarray<double> r2(dataset->inputSize());
		for(size_t j = 0; j < dataset->inputSize(); ++j)
			r2[j] = std::max(r2s[j][i].item<float>(), 0.0f);
		csv::save(config.outputDir/("r2_"+std::to_string(i)+"_importance_"+std::to_string(window)+".csv"), r2, "r2");

		save2dPlot(config.outputDir/("r2_"+std::to_string(i)+"_importance_"+std::to_string(window)+".svg"), indecies, r2, options);
	}
	csv::save(config.outputDir/("loss_importance"+std::to_string(window)+".svg"), loss, "loss importance");
	options.yLabel = "Loss";
	save2dPlot(config.outputDir/("loss_importance"+std::to_string(window)+".svg"), indecies, loss, options);

	return 0;
}

static std::pair<std::valarray<double>, std::valarray<double>> spectraToPlotData(const torch::Tensor& spectra)
{
	torch::Tensor real = torch::real(spectra).to(torch::kFloat64).reshape({-1}).contiguous();
	torch::Tensor imag = torch::imag(spectra).to(torch::kFloat64).reshape({-1}).contiguous();

	std::vector<double> x;
	std::vector<double> y;
	auto realAccessor = real.accessor<double, 1>();
	auto imagAccessor = imag.accessor<double, 1>();
	for(int64_t i = 0; i < real.size(0); ++i)
	{
		if(not std::isfinite(realAccessor[i]) or not std::isfinite(imagAccessor[i]))
			continue;
		x.push_back(realAccessor[i]);
		y.push_back(imagAccessor[i]);
	}
	return {std::valarray<double>(x.data(), x.size()), std::valarray<double>(y.data(), y.size())};
}

template <typename T>
void predictedSpectraPlots(T& dataset, const Config& config, const torch::Tensor& predictions, const torch::Tensor& targets)
{
	std::string modelString = dataset.targetName();
	if(modelString.empty() or modelString == "Unkown")
	{
		Log(Log::WARN)<<"the spectra plots need the dataset to name the circuit its targets describe, "<<config.fileName<<" does not, skipping them";
		return;
	}

	std::shared_ptr<eis::Model> model;
	std::shared_ptr<torch::CompilationUnit> modelScript;
	try
	{
		model = std::make_shared<eis::Model>(modelString);
		modelScript = compileModel(*model);
	}
	catch(const std::exception& error)
	{
		Log(Log::WARN)<<"the circuit "<<modelString<<" can not be compiled, skipping the spectra plots: "<<error.what();
		return;
	}

	if(predictions.size(1) != static_cast<int64_t>(model->getParameterCount()))
	{
		Log(Log::WARN)<<"the network predicts "<<predictions.size(1)<<" parameters but the circuit "<<modelString<<" has "<<model->getParameterCount()<<", skipping the spectra plots";
		return;
	}

	std::filesystem::path imgDir = config.outputDir/"spectra";
	std::filesystem::create_directories(imgDir);

	// one line per plotted spectra: the circuit with the predicted parameters, then the one with
	// the target parameters, so a plot can be read back as a concrete circuit pair
	std::ofstream ecmFile(imgDir/"ecm.txt");

	const int64_t total = std::min(predictions.size(0), static_cast<int64_t>(dataset.size().value()));
	int64_t written = 0;
	int64_t stride = 1;
	if(dataset.size().value() > 200)
		stride = dataset.size().value() / 200;
	for(int64_t i = 0; i < total; i+=stride)
	{
		std::vector<eis::DataPoint> measured = dataset.rawSpectraAtIndex(static_cast<size_t>(i));
		if(measured.size() < 3)
		{
			Log(Log::WARN)<<"spectra "<<i<<" has only "<<measured.size()<<" points, leaving it out of the spectra plots";
			continue;
		}
		std::sort(measured.begin(), measured.end());

		torch::Tensor omegas;
		torch::Tensor measuredSpectra = eisToComplexTensor(measured, &omegas);
		torch::Tensor simulatedSpectra = runScriptModel(*model, modelScript, predictions[i].reshape({-1}),
				omegas.reshape({omegas.numel(), -1})).reshape({omegas.numel()});

		std::pair<std::valarray<double>, std::valarray<double>> predictedPlot = spectraToPlotData(simulatedSpectra);
		if(predictedPlot.first.size() == 0)
		{
			Log(Log::WARN)<<"the parameters of spectra "<<i<<" simulate no finite impedance, leaving it out of the spectra plots";
			continue;
		}

		PlotOptions options;
		options.type = PLOT_TYPE_LINE;
		options.xLabel = "Re(Z)";
		options.yLabel = "Im(Z)";

		save2dPlot(imgDir/("spectra_"+std::to_string(i)+".png"), {predictedPlot, spectraToPlotData(measuredSpectra)}, options);
		ecmFile<<fittedModelString(modelString, predictions[i])<<" | "<<fittedModelString(modelString, targets[i])<<'\n';
		++written;
	}

	Log(Log::INFO)<<"Wrote "<<written<<" measured against predicted spectra plots to "<<imgDir<<" and the circuit pairs to "<<imgDir/"ecm.txt";
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
	// sequential, predictedSpectraPlots() below indexes the dataset by the position a sample came
	// out of the loader, which a random sampler would shuffle
	auto dataLoader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
			dataset.map(torch::data::transforms::Stack<>()), options);

	ann::regression::TestReturn ret;

	if(config.loss == LOSS_MSE)
	{
		std::unique_ptr<torch::nn::MSELoss> lossMse(new torch::nn::MSELoss(torch::nn::MSELossOptions().reduction(torch::kMean)));
		ret = ann::regression::test(net, *dataLoader, *lossMse, dataset.size().value());
	}
	else if(config.loss == LOSS_SPECTRA)
	{
		if(!dataset.frequencies().has_value())
		{
			Log(Log::ERROR)<<"Dataset must provide a omega range for LOSS_SPECTRA";
			return 1;
		}
		torch::Tensor omega = dataset.frequencies().value();
		RegressionNormalization norm = dataset.getNormalization();
		std::unique_ptr<EisDistanceLoss> lossEis(new EisDistanceLoss(dataset.targetName(), omega, net->getOutputScalars(),
				net->getOutputBiases(), net->getOutputLogMask()));
		ret = ann::regression::test(net, *dataLoader, *lossEis, dataset.size().value());
	}
	else if(config.loss == LOSS_PERMUTATION)
	{
		if(!dataset.frequencies().has_value())
		{
			Log(Log::ERROR)<<"Dataset must provide a omega range for LOSS_PERMUTATION";
			return 1;
		}
		torch::Tensor omega = dataset.frequencies().value();
		std::unique_ptr<PermutationMseLoss> lossPermutation(new PermutationMseLoss(dataset.targetName(), omega, net->getOutputScalars(),
				net->getOutputBiases(), net->getOutputLogMask()));
		std::function<torch::Tensor(torch::Tensor, torch::Tensor)> perParameter =
			[&](torch::Tensor output, torch::Tensor targets)
		{
			return lossPermutation->forwardPerParameter(output, targets);
		};
		ret = ann::regression::test(net, *dataLoader, *lossPermutation, dataset.size().value(), 0, nullptr, &perParameter);
	}
	else if(config.loss == LOSS_COMPOSITE)
	{
		if(!dataset.frequencies().has_value())
		{
			Log(Log::ERROR)<<"Dataset must provide a omega range for LOSS_COMPOSITE";
			return 1;
		}
		torch::Tensor omega = dataset.frequencies().value();
		std::unique_ptr<EisCompositeLoss> lossComposite(new EisCompositeLoss(dataset.targetName(), omega, net->getOutputScalars(),
				net->getOutputBiases(), net->getOutputLogMask()));
		ret = ann::regression::test(net, *dataLoader, *lossComposite, dataset.size().value());
	}
	else if(config.loss == LOSS_SIMPLICITY)
	{
		Log(Log::ERROR)<<"the simplicity loss scores circuit labels, it is only available for the classification datasets ("<<DATASET_LIST<<")";
		return 1;
	}
	else
	{
		Log(Log::ERROR)<<"Unkown or unimplemented loss type specified";
		return 1;
	}

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

		PlotOptions options;
		options.type = PLOT_TYPE_SCATTER;
		options.xlog = true;
		options.ylog = true;
		options.xLabel = "Target";
		options.yLabel = "Prediction";

		save2dPlot(config.outputDir/(net->getOutputLabels()[output]+".svg"), targetsVarr, predictionsVarr, options);

		options.xLabel = net->getOutputLabels()[output];
		options.yLabel = "Error";

		save2dPlot(config.outputDir/(net->getOutputLabels()[output]+"_error.svg"), targetsVarr, error, options);
	}

	Log(Log::INFO)<<"\nLoss: "<<ret.loss;
	Log(Log::INFO)<<"Mean MSE: "<<(ret.mse.sum()/ret.mse.numel()).item<float>();
	Log(Log::INFO)<<"Mean R2: "<<(ret.r2.sum()/ret.r2.numel()).item<float>();

	Log(Log::INFO)<<"\nMSE:\n"<<tensorToString(ret.mse, false);
	if(ret.permutationMse.numel() != 0)
		Log(Log::INFO)<<"Permutation adjusted MSE:\n"<<tensorToString(ret.permutationMse, false);
	Log(Log::INFO)<<"R2:\n"<<tensorToString(ret.r2, false);

	if(config.outputMoreImages)
		predictedSpectraPlots<T>(dataset, config, predictions, targets);

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

			PlotOptions options;
			options.type = PLOT_TYPE_LINE;
			options.xLabel = "Re(Z)";
			options.yLabel = "Im(Z)";
			options.xlog = true;
			options.ylog = true;

			save2dPlot(imgDir/("output_" + std::to_string(i) + ".png"), {output, referance}, options);
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

	PlotOptions pltoptions;
	pltoptions.xLabel = "x";
	pltoptions.yLabel = "y";
	pltoptions.zLabel = "z";
	pltoptions.type = PLOT_TYPE_SCATTER;

	switch(ret.latents.size(1))
	{
		default:
		case 3:

			save3dPlot(config.outputDir/"latent.svg", x, y, z, pltoptions);
			break;
		case 2:
		{
			save2dPlot(config.outputDir/"latent.svg", x, y, pltoptions);
			break;
		}
		case 1:
		{
			for(int64_t i = 0; i < ret.latents.size(0); ++i)
				y[i] = i;
			pltoptions.xLabel = "n";
			pltoptions.yLabel = "x";
			save2dPlot(config.outputDir/"latent.svg", y, x, pltoptions);
			break;
		}
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
