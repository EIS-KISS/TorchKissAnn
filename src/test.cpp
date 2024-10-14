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

	torch::Tensor omegas = torch::logspace(-2, 6, 5);

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
	if(autoloss.numel() != 1 || autoloss.item().toDouble() > 0.01)
		return false;

	std::vector<eis::DataPoint> spectra = model.executeSweep(eis::Range(1e-2, 1e6, 5, true));
	torch::Tensor spectraTensor = eisToComplexTensor(spectra);

	torch::Tensor sanityloss = loss.distance(parameters, spectraTensor);
	Log(Log::INFO)<<__func__<<" Sanity loss: "<<sanityloss;
	if(sanityloss.numel() != 1 || sanityloss.item().toDouble() > 0.01)
		return false;

	torch::Tensor adjloss = loss.forward(parametersAdj, parameters);
	Log(Log::INFO)<<__func__<<" Adj loss (expect high): "<<adjloss;
	if(adjloss.numel() != 1 || adjloss.item().toDouble() < 1)
		return false;

	return true;
}

bool testFit()
{
	const std::string modelString = "r{100}c{1e-5}";
	eis::Model model(modelString);
	torch::Tensor omegas;
	torch::Tensor targetSpectra = eisToComplexTensor(model.executeSweep(eis::Range(1, 1e6, 5)), &omegas);

	Log(Log::INFO)<<__func__<<"targetSpectra:\n"<<targetSpectra<<"\nomegas:\n"<<omegas;

	std::pair<torch::Tensor, torch::Tensor> fited = eisFit(targetSpectra, omegas, modelString);

	Log(Log::INFO)<<__func__<<"Result:\n"<<fited.first<<"\nremaining loss:\n"<<fited.second;
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

	/*//testEisGeneratorDataset();
	//reexportDirDataset();

	testLinearToComplex();
	testEisDistanceLoss();
	//testParaDataset();
	testEisScript();*/
	//testEisDistanceLoss();
	//testFit();
	testScriptnet();

	free_device();
	return 0;
}
