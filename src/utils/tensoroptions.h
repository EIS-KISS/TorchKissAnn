#pragma once
#include <torch/torch.h>

template <typename V>
inline torch::TensorOptions tensorOptCpu(bool grad = true)
{
	static_assert(std::is_same<V, float>::value || std::is_same<V, double>::value,
				  "This function can only be passed double or float types");
	torch::TensorOptions options;
	if constexpr(std::is_same<V, float>::value)
		options = options.dtype(torch::kFloat32);
	else
		options = options.dtype(torch::kFloat64);
	options = options.layout(torch::kStrided);
	options = options.device(torch::kCPU);
	options = options.requires_grad(grad);
	return options;
}

template <typename V>
inline torch::TensorOptions tensorOptCplxCpu(bool grad = true)
{
	static_assert(std::is_same<V, float>::value || std::is_same<V, double>::value,
				  "This function can only be passed double or float types");
	torch::TensorOptions options;
	if constexpr(std::is_same<V, float>::value)
		options = options.dtype(torch::kComplexFloat);
	else
		options = options.dtype(torch::kComplexDouble);
	options = options.layout(torch::kStrided);
	options = options.device(torch::kCPU);
	options = options.requires_grad(grad);
	return options;
}
