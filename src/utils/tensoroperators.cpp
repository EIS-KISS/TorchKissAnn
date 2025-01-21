//
// TorchKissAnn - A collection of tools to train various types of Machine learning
// algorithms on various types of EIS data
// Copyright (C) 2025 Carl Klemm <carl@uvos.xyz>
//
// This file is part of TorchKissAnn.
//
// TorchKissAnn is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// TorchKissAnn is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with TorchKissAnn.  If not, see <http://www.gnu.org/licenses/>.
//

#include "tensoroperators.h"

#include <c10/core/ScalarType.h>
#include <fstream>
#include <sstream>
#include <torch/types.h>
#include <torch/script.h>

#include "log.h"
#include "save.h"

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

bool saveTensorForPytorch(const std::filesystem::path& path, const torch::Tensor& tensor)
{
	std::fstream file;
	file.open(path, std::ios_base::out | std::ios_base::binary);
	if(!file.is_open())
	{
		Log(Log::ERROR)<<"can not open "<<path<<" for writing\n";
		return false;
	}

	std::vector<char> tensorBinary = torch::jit::pickle_save(tensor);
	file.write(tensorBinary.data(), tensorBinary.size());
	file.close();
	return true;
}

torch::Tensor linearToComplex(torch::Tensor& in)
{
	assert((in.dtype() == torch::kFloat32 || in.dtype() == torch::kFloat64) && in.numel() % 2 == 0);
	return torch::complex(torch::slice(in, 0, 0, in.numel()/2), torch::slice(in, 0, in.numel()/2, in.numel()));
}

torch::Tensor complexToLinear(torch::Tensor& in)
{
	assert(in.dtype() == torch::kComplexFloat || in.dtype() == torch::kComplexDouble);
	torch::Tensor imag = torch::imag(in);
	imag = imag.reshape({imag.numel()});
	torch::Tensor real = torch::real(in);
	real = real.reshape({real.numel()});
	torch::Tensor out = torch::cat({real, imag}, 0);
	assert(out.sizes().size() == 1);
	assert(out.size(0) == in.numel()*2);
	return out;
}

std::string tensorToString(const torch::Tensor& tensor)
{
	std::stringstream ss;
	ss<<std::scientific;

	if(tensor.dim() == 2)
	{
		if(tensor.dtype() == torch::kFloat32)
			csv::save2dTensorToCsv<float>(tensor, ss);
		else
			csv::save2dTensorToCsv<int64_t>(tensor, ss);
	}
	else if(tensor.dim() == 1)
	{
		if(tensor.dtype() == torch::kFloat32)
			csv::save1dTensorToCsv<float>(tensor, ss);
		else
			csv::save1dTensorToCsv<int64_t>(tensor, ss);
	}
	else
	{
		Log(Log::ERROR)<<"cant convert "<<tensor.dim()<<" dimentional tensor to string\n";
		return "";
	}
	return ss.str();
}



torch::Tensor safeConcat(std::initializer_list<torch::Tensor> list, int64_t dim)
{
	torch::Tensor output;
	for(const torch::Tensor& in : list)
	{
		if(in.numel() == 0)
			continue;

		if(output.numel() == 0)
			output = in;
		else
			output = torch::cat({output, in}, dim);
	}
	return output;
}
