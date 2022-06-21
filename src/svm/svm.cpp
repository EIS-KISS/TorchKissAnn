#include "svm.h"
#include <libsvm/svm.h>
#include "utils/log.h"
#include "svmloader.h"
#include "data/generators/eisgendata.h"
#include "constants.h"

using namespace svm;

static void discardStr(const char* str)
{
	(void)str;
}

static svm_parameter getSvmParameters()
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

void svm::cross()
{
	svm_set_print_string_function(&discardStr);
	EisGeneratorDataset* eisData = EisGeneratorDataset::getNeisGeneratedDataset(DATA_WIDTH, 20e4);
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

void svm::train(bool oneClass)
{
	EisGeneratorDataset* eisData = EisGeneratorDataset::getNeisGeneratedDataset(DATA_WIDTH, EisGeneratorDataset::DEFAULT_EXAMPLE_COUNT);

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

	EisGeneratorDataset* eisDataAlt = EisGeneratorDataset::getNeisGeneratedDataset(DATA_WIDTH, 20e4);
	test<EisGeneratorDataset>(model, eisDataAlt, oneClass);

	constexpr char fileName[] = "./svm.vos";
	constexpr char fileNameOneClass[] = "./svmFilter.vos";
	svm_save_model(oneClass ? fileNameOneClass : fileName, model);
	svm_free_and_destroy_model(&model);
	Log(Log::INFO)<<"Saved svm model as "<<(oneClass ? fileNameOneClass : fileName);
}

bool svm::filter(torch::Tensor input, std::string filenName)
{
	svm_model* model = svm_load_model(filenName.c_str());
	if(!model)
	{
		Log(Log::ERROR)<<"could not load SVM model from "<<filenName;
		exit(-1);
	}

	svm_node* data = SvmLoader<EisGeneratorDataset>::tensorToNewSvmNodeArray(input);

	Log(Log::DEBUG)<<"svm type: "<<svm_get_svm_type(model);
	if(svm_get_svm_type(model) != ONE_CLASS)
	{
		Log(Log::ERROR)<<"Svm model "<<filenName<<" must be of type "<<ONE_CLASS<<" but is of type "<<svm_get_svm_type(model);
		exit(-1);
	}

	double prediction = svm_predict(model, data);
	Log(Log::DEBUG)<<"filter prediction: "<<prediction;
	svm_free_and_destroy_model(&model);
	delete[] data;
	return prediction > 0;
}


torch::Tensor svm::use(torch::Tensor input, std::string filenName)
{
	svm_model* model = svm_load_model(filenName.c_str());
	if(!model)
	{
		Log(Log::ERROR)<<"could not load SVM model from "<<filenName;
		exit(-1);
	}

	svm_node* data = SvmLoader<EisGeneratorDataset>::tensorToNewSvmNodeArray(input);

	Log(Log::DEBUG)<<"svm type: "<<svm_get_svm_type(model);
	if(svm_get_svm_type(model) != C_SVC)
	{
		Log(Log::ERROR)<<"Svm model "<<filenName<<" must be of type "<<C_SVC<<" but is of type "<<svm_get_svm_type(model);
		exit(-1);
	}

	double* estimates = new double[svm_get_nr_class(model)];
	svm_predict_probability(model, data, estimates);

	torch::Tensor output = torch::empty(svm_get_nr_class(model));

	for(int64_t i = 0; i < output.size(0); ++i)
		output[i] = estimates[i];

	svm_free_and_destroy_model(&model);
	delete[] data;
	delete[] estimates;
	return output;
}
