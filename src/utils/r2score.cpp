#include "r2score.h"

torch::Tensor r2score(const torch::Tensor& prediction, const torch::Tensor& truth)
{
	torch::Tensor means = truth.mean(0).repeat({truth.size(0), 1});
	torch::Tensor suqareSum = torch::sum(torch::pow(truth-prediction, 2), 0);
	torch::Tensor squareSumMean = torch::sum(torch::pow(truth-means, 2), 0);
	return 1-suqareSum/squareSumMean;
}
