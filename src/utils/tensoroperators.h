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
#include <cstdint>
#include <functional>
#include <filesystem>
#include <initializer_list>

template <typename V>
bool checkTorchType(const torch::Tensor& tensor)
{
	static_assert(std::is_same<V, float>::value ||
		std::is_same<V, double>::value ||
		std::is_same<V, int64_t>::value ||
		std::is_same<V, int32_t>::value ||
		std::is_same<V, int8_t>::value ||
		std::is_same<V, std::complex<float>>::value ||
		std::is_same<V, std::complex<double>>::value,
				  "This function dose not work with this type");
	if constexpr(std::is_same<V, float>::value)
		return tensor.dtype() == torch::kFloat32;
	else if constexpr(std::is_same<V, double>::value)
		return tensor.dtype() == torch::kFloat64;
	else if constexpr(std::is_same<V, int64_t>::value)
		return tensor.dtype() == torch::kInt64;
	else if constexpr(std::is_same<V, int32_t>::value)
		return tensor.dtype() == torch::kInt32;
	else if constexpr(std::is_same<V, int8_t>::value)
		return tensor.dtype() == torch::kInt8;
	else if constexpr(std::is_same<V, std::complex<float>>::value)
		return tensor.dtype() == torch::kComplexFloat;
	else if constexpr(std::is_same<V, std::complex<double>>::value)
		return tensor.dtype() == torch::kComplexDouble;
}

torch::Tensor toeplitz(const torch::Tensor& a, const torch::Tensor b);

torch::Tensor fnToTensor(std::function<double(double)> fn, double start, double stop, size_t size);

double tensorSimpleIntegrate(torch::Tensor in, double length);

double integrateFn(std::function<double(double)> fn, double start, double stop, size_t steps = 100, double *prev = nullptr);

torch::Tensor unargmax(const torch::Tensor& in, int64_t size);

torch::Tensor complexToLinear(torch::Tensor& tensor);
torch::Tensor linearToComplex(torch::Tensor& tensor);

std::string tensorToString(const torch::Tensor& tensor);

torch::Tensor safeConcat(std::initializer_list<torch::Tensor> list, int64_t dim);
