#pragma once
#include <c10/core/ScalarType.h>
#include <torch/torch.h>

#include "globals.h"

template <typename V>
inline torch::TensorOptions tensorOptCpu(bool grad = true)
{
	static_assert(std::is_same<V, float>::value || std::is_same<V, double>::value || std::is_same<V, long>::value,
				  "This function can only be passed double, float or long types");
	torch::TensorOptions options;
	if constexpr(std::is_same<V, float>::value)
		options = options.dtype(torch::kFloat32);
	else if constexpr(std::is_same<V, double>::value)
		options = options.dtype(torch::kFloat64);
	else
		options = options.dtype(torch::kLong);
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

template <typename V>
inline torch::TensorOptions tensorOptDevice(bool grad = true)
{
	return tensorOptCpu<V>().device(*offload_device);
}
