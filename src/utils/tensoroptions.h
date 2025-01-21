/* * TorchKissAnn - A collection of tools to train various types of Machine learning
 * algorithms on various types of EIS data
 * Copyright (C) 2025 Carl Klemm <carl@uvos.xyz>
 *
 * This file is part of TorchKissAnn.
 *
 * TorchKissAnn is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * TorchKissAnn is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with TorchKissAnn.  If not, see <http://www.gnu.org/licenses/>.
 */

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
