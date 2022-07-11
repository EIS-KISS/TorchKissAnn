#include "drt.h"
#include <functional>
#include <cmath>
#include <cstdint>

#include "../utils/tensoroperators.h"

static torch::Tensor generateMatrixA(const torch::Tensor& freqs, double eps, std::function<double(double)> fn, bool imPart)
{
	torch::Tensor diff = torch::log(1/freqs);
	double mean = torch::mean(diff).item().to<float>();
	double stdDeviation = torch::std(diff).item().to<float>();

	bool useTopiltz = stdDeviation/mean < 0.01;

	torch::Tensor inter;

	if(useTopiltz)
	{

	}
}

void doDrt(const std::vector<eis::DataPoint>& data, torch::Tensor& drtOut)
{
	torch::Tensor freqs;
	torch::Tensor input = eisToTorch(data, &freqs);
	torch::Tensor tau = 1/freqs;

	input.reshape({input.numel()/2, 2});

}
