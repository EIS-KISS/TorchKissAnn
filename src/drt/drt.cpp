#include "drt.h"
#include <functional>
#include <cmath>
#include <cstdint>

#include "../utils/tensoroperators.h"

static double matrixAElement(std::function<double(double)>& fn, double freq, double tau, bool imPart)
{
	double alpha = 2*M_PI*freq*tau;
	std::function<double(double)> fnFInteg;
	if(imPart)
		fnFInteg = [alpha, fn](double x) -> double {return alpha/(1/std::exp(x)+std::pow(alpha, 2)*std::exp(x))*fn(x);};
	else
		fnFInteg = [alpha, fn](double x) -> double {return 1/(1+std::pow(alpha, 2)*std::exp(2*x))*fn(x);};
	return integrateFn(fnFInteg, -50, 50);
}

static torch::Tensor generateMatrixA(const torch::Tensor& freqs, const torch::Tensor& taus, std::function<double(double)> fn, bool imPart)
{
	torch::Tensor diff = torch::log(1/freqs);
	double mean = torch::mean(diff).item().to<float>();
	double stdDeviation = torch::std(diff).item().to<float>();

	bool useTopiltz = stdDeviation/mean < 0.01 && taus.numel() == freqs.numel();

	torch::Tensor inter;

	torch::Tensor out = torch::zeros({freqs.numel(), taus.numel()}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
	auto outA = out.accessor<float, 2>();

	auto Fa = freqs.accessor<float, 1>();
	auto Ta = taus.accessor<float, 1>();

	if(useTopiltz)
	{
		torch::Tensor R = torch::zeros({taus.numel()}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
		torch::Tensor C = torch::zeros({freqs.numel()}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
		auto Ra = R.accessor<float, 1>();
		auto Ca = C.accessor<float, 1>();

		for(int64_t i = 0; i < Ca.size(0); ++i)
		{
			Ca[i] = matrixAElement(fn, Fa[i], Ta[0], imPart);
		}

		for(int64_t i = 0; i < Ra.size(0); ++i)
		{
			Ra[i] = matrixAElement(fn, Fa[0], Ta[i], imPart);
		}

		out = toeplitz(C, R);
	}
	else
	{
		for(int64_t i = 0; i < freqs.numel(); ++i)
		{
			for(int64_t j = 0; j < taus.numel(); ++j)
				outA[i][j] = matrixAElement(fn, Fa[i], Ta[j], imPart);
		}
	}

	return out;
}

//TODO: works for gaus only
static double innerProd(double freqA, double freqB, double epsilon)
{
	double a = epsilon*std::log(freqA/freqB);

	return -epsilon*(-1*std::pow(a, 2))*std::exp(0-(std::pow(a, 2)/2))*std::sqrt(M_PI/2);
}

torch::Tensor generateMatrixM(torch::Tensor taus, double epsilon, std::function<double(double)> fn)
{
	torch::Tensor recoverdFreq = 1/taus;
	auto recoverdFreqA = recoverdFreq.accessor<float, 1>();
	torch::Tensor out = torch::zeros({taus.numel(), taus.numel()}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

	double stdDiff = torch::std(torch::diff(torch::log(taus))).item().to<double>();
	double meanDiff = torch::mean(torch::diff(torch::log(taus))).item().to<double>();

	bool useTopiltz = stdDiff/meanDiff < 0.01;

	if(useTopiltz)
	{
		torch::Tensor R = torch::zeros({taus.numel()}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
		torch::Tensor C = torch::zeros({taus.numel()}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
		auto Ra = R.accessor<float, 1>();
		auto Ca = C.accessor<float, 1>();

		for(int64_t i = 0; i < Ca.size(0); ++i)
		{
			Ca[i] = innerProd(recoverdFreqA[0], recoverdFreqA[i], epsilon);
		}

		for(int64_t i = 0; i < Ra.size(0); ++i)
		{
			Ra[i] = innerProd(recoverdFreqA[i], recoverdFreqA[0], epsilon);
		}

		out = toeplitz(C, R);
	}
	else
	{
		auto outA = out.accessor<float, 2>();
		for(int64_t i = 0; i < out.size(0); ++i)
		{
			for(int64_t j = 0; j < out.size(1); ++j)
			{
				outA[i][j] = innerProd(recoverdFreqA[i], recoverdFreqA[j], epsilon);
			}
		}
	}

	return out;
}

double discretizationFn(double x, double epsilon)
{
	return std::exp(-std::pow(epsilon*x, 2));
}

void doDrt(const std::vector<eis::DataPoint>& data, torch::Tensor& drtOut)
{
	torch::Tensor freqs;
	torch::Tensor input = eisToTorch(data, &freqs);
	input.reshape({input.numel()/2, 2});

	torch::Tensor real = input.select(0, 0);
	torch::Tensor imag = input.select(0, 1);

	torch::Tensor taus = 1/freqs;
	double epsilon = 0.5;

	torch::Tensor aRe = generateMatrixA(freqs, taus, [epsilon](double x) -> double {return discretizationFn(x, epsilon);}, false);
	torch::Tensor aIm = generateMatrixA(freqs, taus, [epsilon](double x) -> double {return discretizationFn(x, epsilon);}, true);

	torch::Tensor M = generateMatrixM(taus, epsilon, [epsilon](double x) -> double {return discretizationFn(x, epsilon);});
}
