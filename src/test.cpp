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

#include <asm-generic/errno.h>
#include <cstddef>
#include <cstdint>
#include <kisstype/type.h>
#include <iostream>
#include <libsvm/svm.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/optim.h>
#include <filesystem>

#include "ann/scriptnet.h"
#include "data/eistotorch.h"
#include "log.h"
#include "tensoroperators.h"
#include "ann/simplenet.h"
#include "data/loaders/dirloader.h"
#include "tensoroptions.h"
#include "loss/eisdistanceloss.h"
#include "modelscript.h"
#include "fit/fit.h"
#include "tokenize.h"

template<typename Dataset>
void testLoader(torch::data::datasets::Dataset<Dataset, torch::data::Example<torch::Tensor, torch::Tensor>>* data)
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

bool cmpDouble(double a, double b)
{
	double diff = std::fabs(a-b);
	return diff < 0.0001;
}

/*void saveDataset()
{
	EisGeneratorDataset* testDataset = EisGeneratorDataset::getNeisGeneratedDataset(DATA_WIDTH, 1e4);
	ClassExtractorDataset<EisGeneratorDataset> extractor(testDataset, 5);
	saveDataset<ClassExtractorDataset<EisGeneratorDataset>>("./testdatasetexport", &extractor);
}*/

void testfs(const std::string& str, const std::string& prefix)
{
	const std::filesystem::path directoryPath{str};
	for(const std::filesystem::directory_entry& dirent : std::filesystem::directory_iterator{directoryPath})
	{
		std::string fileName = dirent.path().filename();
		std::cout<<fileName;
		if(dirent.is_regular_file() && fileName.size() > prefix.size() && fileName.find(prefix) == 0)
		{
			fileName.erase(fileName.begin(), fileName.begin()+prefix.size());
			std::vector<std::string> tokens = tokenize(fileName, '_');
			size_t classNum = std::stoul(tokens[0]);

			std::cout<<" is "<<classNum<<'\n';
		}
		else
		{
			std::cout<<" is nothing\n";
		}
	}
}

bool reexportDirDataset()
{
	EisDirDataset dataset("realDataset/train");
	Log(Log::INFO)<<__func__<<" size: "<<dataset.size().value();
	Log(Log::INFO)<<__func__<<" classes: "<<dataset.outputSize();
	torch::Tensor classCounts = dataset.classCounts();
	for(size_t i = 0; i < dataset.outputSize(); ++i)
		Log(Log::INFO)<<__func__<<" class "<<i<<", "<<classCounts[i].item().to<int64_t>()<<" examples: "<<dataset.outputName(i);
	/*bool ret = saveDataset<EisDirDataset>("datasetexport", &dataset);
	if(!ret)
		Log(Log::ERROR)<<__func__<<" could not save dataset";*/
	return true;
}

void testModelGrad()
{
	ann::SimpleNet net;
	//Log(Log::DEBUG)<<net[0].weights
}

bool testLinearToComplex()
{
	torch::Tensor data = torch::randn({100}, tensorOptCpu<fvalue>());
	torch::Tensor complex = linearToComplex(data);
	if(complex.numel() != data.numel()/2)
		return false;
	torch::Tensor linear = complexToLinear(complex);
	if(torch::sum(data-linear).item().toDouble() > 0.001)
		return false;
	return true;
}

bool testEisScript()
{
	eis::Model model("r{100}-r{100}c{1e-4}");
	std::shared_ptr<torch::CompilationUnit> modelScript = compileModel(model);

	torch::Tensor omegas = torch::logspace(-2, 6, 5);

	std::vector<fvalue> paramVect = model.getFlatParameters();
	torch::Tensor parameters = fvalueVectorToTensor(paramVect).set_requires_grad(false);
	parameters = parameters.repeat({2, 1}).t();

	torch::Tensor spectra = runScriptModel(model, modelScript, parameters, omegas.reshape({omegas.numel(), -1})).t();

	Log(Log::INFO)<<"parameters\n"<<parameters;
	Log(Log::INFO)<<"spectra\n"<<spectra;

	return true;
}

bool testEisDistanceLoss()
{
	eis::Model model("r{100}-r{100}c{1e-4}");
	model.compile();

	torch::Tensor omegas = torch::logspace(-2, 4, 7);

	EisDistanceLoss loss(model, omegas);
	std::vector<fvalue> paramVect = model.getFlatParameters();
	Log(Log::INFO)<<"Parameters: "<<paramVect.size();
	if(paramVect.size() != 3)
	{
		Log(Log::ERROR)<<"wrong nummber of parameters! should be 3";
		return false;
	}
	torch::Tensor parameters = fvalueVectorToTensor(paramVect).set_requires_grad(false);

	if(parameters.numel() != static_cast<int64_t>(paramVect.size()))
	{
		Log(Log::ERROR)<<"fvalueVectorToTensor returned a tensor with numel "<<parameters.numel()
		<<" for a input vector of size "<<paramVect.size();
		return false;
	}

	torch::Tensor parametersAdj = parameters.clone();
	parametersAdj[2] = 2e-4;

	Log(Log::INFO)<<"Parameters:\n"<<parameters<<"\nParametersAdj:\n"<<parametersAdj;

	torch::Tensor autoloss = loss.forward(parameters, parameters);
	Log(Log::INFO)<<__func__<<" Autoloss: "<<autoloss;
	if(autoloss.numel() != 1 || autoloss.item().toDouble() > 1e-6)
	{
		Log(Log::ERROR)<<"the loss between identical parameters must be zero, is "<<autoloss;
		return false;
	}

	std::vector<eis::DataPoint> spectra = model.executeSweep(eis::Range(1e-2, 1e4, 7, true));
	torch::Tensor spectraTensor = eisToComplexTensor(spectra);

	torch::Tensor sanityloss = loss.distance(parameters, spectraTensor);
	Log(Log::INFO)<<__func__<<" Sanity loss: "<<sanityloss;
	if(sanityloss.numel() != 1 || sanityloss.item().toDouble() > 1e-6)
	{
		Log(Log::ERROR)<<"the torchscript model and executeSweep disagree, loss is "<<sanityloss;
		return false;
	}

	torch::Tensor adjloss = loss.forward(parametersAdj, parameters);
	Log(Log::INFO)<<__func__<<" Adj loss (expect high): "<<adjloss;
	if(adjloss.numel() != 1 || adjloss.item().toDouble() < 1e-3)
	{
		Log(Log::ERROR)<<"doubling the capacitance is not visible in the loss, loss is "<<adjloss;
		return false;
	}

	// the loss is a mean, so a batch of identical samples scores like a single sample
	torch::Tensor batch = parameters.repeat({4, 1});
	torch::Tensor batchAuto = loss.forward(batch, batch);
	Log(Log::INFO)<<__func__<<" Batch autoloss: "<<batchAuto;
	if(batchAuto.numel() != 1 || batchAuto.item().toDouble() > 1e-6)
	{
		Log(Log::ERROR)<<"the loss of a batch of identical samples must be zero, is "<<batchAuto;
		return false;
	}

	// one wrong sample in a batch of four contributes a quarter of its loss on its own
	torch::Tensor batchAdj = batch.clone();
	batchAdj[1] = parametersAdj;
	torch::Tensor batchAdjLoss = loss.forward(batchAdj, batch);
	Log(Log::INFO)<<__func__<<" Batch adj loss (expect "<<adjloss.item().toDouble()/4<<"): "<<batchAdjLoss;
	if(std::fabs(batchAdjLoss.item().toDouble() - adjloss.item().toDouble()/4) > adjloss.item().toDouble()/40)
	{
		Log(Log::ERROR)<<"the loss is not the mean over the batch, is "<<batchAdjLoss
			<<" for a single sample loss of "<<adjloss;
		return false;
	}

	// doubling the batch size must not change the loss
	torch::Tensor batchAdj8Loss = loss.forward(batchAdj.repeat({2, 1}), batch.repeat({2, 1}));
	Log(Log::INFO)<<__func__<<" Batch of eight adj loss: "<<batchAdj8Loss;
	if(std::fabs(batchAdj8Loss.item().toDouble() - batchAdjLoss.item().toDouble()) > batchAdjLoss.item().toDouble()/10)
	{
		Log(Log::ERROR)<<"the loss scales with the batch size, "<<batchAdjLoss<<" vs "<<batchAdj8Loss;
		return false;
	}

	// a fixed relative error costs the same at every frequency: with a plain sum of the
	// impedances the lowest frequency of a spectra spanning decades outweighs all others
	torch::Tensor lowOnly = spectraTensor.clone();
	lowOnly[0] = lowOnly[0]*1.1;
	torch::Tensor highOnly = spectraTensor.clone();
	highOnly[spectraTensor.numel()-1] = highOnly[spectraTensor.numel()-1]*1.1;
	torch::Tensor lowLoss = loss.distance(parameters, lowOnly);
	torch::Tensor highLoss = loss.distance(parameters, highOnly);
	Log(Log::INFO)<<__func__<<" 10% error at the lowest frequency: "<<lowLoss<<" at the highest: "<<highLoss;
	if(lowLoss.item().toDouble() > highLoss.item().toDouble()*10 ||
			highLoss.item().toDouble() > lowLoss.item().toDouble()*10)
	{
		Log(Log::ERROR)<<"a single frequency dominates the loss, "<<lowLoss<<" vs "<<highLoss;
		return false;
	}

	// the loss is relative to the impedance of a sample: the same relative error on a circuit
	// with a thousand times the impedance must score the same
	torch::Tensor scaled = parameters.clone();
	scaled[0] = scaled[0]*1000;
	scaled[1] = scaled[1]*1000;
	scaled[2] = scaled[2]/1000;
	torch::Tensor scaledAdj = scaled.clone();
	scaledAdj[2] = scaledAdj[2]*2;
	torch::Tensor scaledLoss = loss.forward(scaledAdj, scaled);
	Log(Log::INFO)<<__func__<<" Adj loss on a circuit with 1000x the impedance: "<<scaledLoss;
	if(std::fabs(scaledLoss.item().toDouble() - adjloss.item().toDouble()) > adjloss.item().toDouble()/10)
	{
		Log(Log::ERROR)<<"the loss depends on the absolute impedance, "<<scaledLoss<<" vs "<<adjloss;
		return false;
	}

	// the trainer hands the loss normalised parameters, the extreme sample of a dataset
	// normalises to exactly zero, which would be a division by zero inside the model script
	torch::Tensor scalar = torch::tensor({200.0f, 200.0f, 2e-4f});
	EisDistanceLoss normalisedLoss(model, omegas, scalar, parameters.clone());
	torch::Tensor normalisedBatch = torch::zeros({4, 3}, parameters.options());
	torch::Tensor normalisedAuto = normalisedLoss.forward(normalisedBatch, normalisedBatch);
	Log(Log::INFO)<<__func__<<" Loss of a batch of normalised zeros: "<<normalisedAuto;
	if(!torch::isfinite(normalisedAuto).all().item().to<bool>() || normalisedAuto.item().toDouble() > 1e-6)
	{
		Log(Log::ERROR)<<"normalised parameters are not undone correctly, loss is "<<normalisedAuto;
		return false;
	}

	// the gradient must be finite, in particular for a prediction that is exactly right
	torch::Tensor gradParams = parameters.clone().set_requires_grad(true);
	loss.forward(gradParams.reshape({1, -1}), parameters.reshape({1, -1})).backward();
	if(!torch::isfinite(gradParams.grad()).all().item().to<bool>())
	{
		Log(Log::ERROR)<<"the gradient of a perfect prediction is not finite:\n"<<gradParams.grad();
		return false;
	}

	// and it must carry information about the parameters when the prediction is wrong
	torch::Tensor gradParamsAdj = parametersAdj.clone().set_requires_grad(true);
	loss.forward(gradParamsAdj.reshape({1, -1}), parameters.reshape({1, -1})).backward();
	Log(Log::INFO)<<__func__<<" Gradient of the adjusted parameters:\n"<<gradParamsAdj.grad();
	if(!torch::isfinite(gradParamsAdj.grad()).all().item().to<bool>() ||
			torch::sum(torch::abs(gradParamsAdj.grad())).item().toDouble() == 0)
	{
		Log(Log::ERROR)<<"the gradient of a wrong prediction carries no information:\n"<<gradParamsAdj.grad();
		return false;
	}

	return true;
}

bool testFit()
{
	const std::string modelString = "r{100}c{1e-5}";
	eis::Model model(modelString);
	torch::Tensor omegas;
	torch::Tensor targetSpectra = eisToComplexTensor(model.executeSweep(eis::Range(1, 1e6, 5)), &omegas);

	Log(Log::INFO)<<__func__<<"targetSpectra:\n"<<targetSpectra<<"\nomegas:\n"<<omegas;

	FitResult fited = eisFit(targetSpectra, omegas, modelString);

	Log(Log::INFO)<<__func__<<"Result:\n"<<fited.parameters<<"\nremaining loss:\n"<<fited.chiSquared;
	return true;
}


bool testScriptnet()
{
	ann::SimpleNet net(100, 6, 4, 3, true);
	net.eval();
	torch::Tensor input = torch::randn({1, 100});
	torch::Tensor output = net.forward(input);
	Log(Log::INFO)<<"SimpleNet sizes: "<< output.sizes();
	Log(Log::INFO)<<output;

	ann::ScriptNet scriptNet("../TorchScripts/simplenet100-6.pt", false, 100, 6);

	output = scriptNet.forward(input);
	Log(Log::INFO)<<"ScriptNet sizes: "<< output.sizes();
	Log(Log::INFO)<<output;

	return true;
}

int main(int argc, char** argv)
{
	Log::level = Log::DEBUG;
	choose_device(true);

	bool success = true;

	success &= testLinearToComplex();
	success &= testEisScript();
	success &= testEisDistanceLoss();

	// requires a torchscript network that is not part of the repository
	if(std::filesystem::exists("../TorchScripts/simplenet100-6.pt"))
		success &= testScriptnet();
	else
		Log(Log::INFO)<<"Skipping the scriptnet test, ../TorchScripts/simplenet100-6.pt does not exist";

	free_device();
	return success ? 0 : 1;
}
