#include "svmloader.h"

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
