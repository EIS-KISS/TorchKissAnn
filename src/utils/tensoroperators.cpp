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
