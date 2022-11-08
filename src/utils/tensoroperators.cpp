#include "tensoroperators.h"
#include "log.h"

torch::Tensor toeplitz(const torch::Tensor& b, const torch::Tensor a)
{
	torch::Tensor indeciesB = torch::linspace(1, b.numel()-1, b.numel()-1, torch::TensorOptions().dtype(torch::kInt32));
	torch::Tensor line = torch::cat({a.reshape({1, a.numel()}), torch::index_select(b.reshape({1, b.numel()}), 1, indeciesB).flip(1)}, 1);
	torch::Tensor output = line;
	for(int64_t i = 1; i < b.numel(); ++i)
		output = torch::cat({output, torch::roll(line, i)}, 0);

	torch::Tensor indeciesCols = torch::linspace(0, a.numel()-1, a.numel(), torch::TensorOptions().dtype(torch::kInt32));
	return torch::index_select(output, 1, indeciesCols);
}

torch::Tensor fnToTensor(std::function<double(double)> fn, double start, double stop, size_t size)
{
	torch::Tensor out = torch::empty({static_cast<int64_t>(size)}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
	auto outA = out.accessor<float, 1>();

	double step = (stop-start)/(size-1);

	for(size_t i = 0; i < size; ++i)
		outA[i] = fn(start+i*step);
	return out;
}

double tensorSimpleIntegrate(torch::Tensor in, double length)
{
	return (torch::mean(in).item().to<double>())*length;
}

double integrateFn(std::function<double(double)> fn, double start, double stop, size_t steps, double *prev)
{
	double resH =  prev ? *prev : tensorSimpleIntegrate(fnToTensor(fn, start, stop, steps/2), stop-start);
	double res = tensorSimpleIntegrate(fnToTensor(fn, start, stop, steps), stop-start);

	double relDiff = res/resH;
	if((relDiff > 0.99 && relDiff < 1.01) || steps > 3e6)
		return res;
	else
		return integrateFn(fn, start, stop, steps*2, &res);
}

torch::Tensor unargmax(const torch::Tensor& in, int64_t size)
{
	assert(size >= 0);
	assert(in.dim() == 1);
	torch::Tensor multiClassTargets = torch::zeros({in.size(0), size}, in.options());

	for(int i = 0; i < in.size(0); ++i)
	{
		multiClassTargets[i][in[i].template item<int64_t>()] = 1;
	}
	return multiClassTargets;
}

