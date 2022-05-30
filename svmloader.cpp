#include "svmloader.h"

void printSvmNode(const svm_node* nodeArray)
{
	for(size_t i = 0; nodeArray[i].index >= 0 ; ++i)
	{
		Log(Log::DEBUG, false)<<nodeArray[i].index<<":"<<nodeArray[i].value<<" ";
	}
	Log(Log::DEBUG)<<"";
}

void printSvmProblem(svm_problem problem)
{
	for(int i = 0; i < problem.l; ++i)
	{
		Log(Log::DEBUG)<<"line "<<i;
		Log(Log::DEBUG)<<problem.y[i]<<"---";
		for(int k = 0; problem.x[i][k].index >= 0; ++k)
		{
			int index = problem.x[i][k].index;
			double value = problem.x[i][k].value;
			Log(Log::DEBUG, false)<<index<<":"<<value<<" ";
		}
		Log(Log::DEBUG)<<"";
	}
}

void svmPrintFunction(const char* str)
{
	Log(Log::DEBUG, false)<<str;
}

void setSvmWeights(svm_parameter* params, const torch::Tensor& weigts)
{
	params->nr_weight = weigts.numel();
	params->weight_label = new int[params->nr_weight];
	params->weight = new double[params->nr_weight];
	//weigts = weigts.reshape({1, weigts.numel()});
	for(int i = 0; i < params->nr_weight; ++i)
	{
		params->weight_label[i] = i;
		params->weight[i] = 1/weigts[i].item().to<float>();
	}
}
