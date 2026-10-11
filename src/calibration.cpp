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

#include "calibration.h"

#include <cmath>
#include <torch/torch.h>

#include "globals.h"
#include "utils/log.h"

calibration::Accumulator::Accumulator(const std::shared_ptr<ann::Net>& net) :
	network(net)
{
}

void calibration::Accumulator::add(const torch::Tensor& inputs, const torch::Tensor& targets)
{
	torch::NoGradGuard noGrad;
	network->eval();

	torch::Tensor output = network->forward(network->normalizeInput(inputs.to(*offload_device)));
	torch::Tensor normalizedTargets = network->normalizeTarget(targets.to(*offload_device));

	torch::Tensor batchOut = output.to(torch::kFloat64).sum(0).to(torch::kCPU);
	torch::Tensor batchTarget = normalizedTargets.to(torch::kFloat64).sum(0).to(torch::kCPU);
	double n = static_cast<double>(output.size(0));

	if(!outputSum.defined())
	{
		outputSum = torch::zeros({output.size(1)}, torch::TensorOptions().dtype(torch::kFloat64));
		targetSum = torch::zeros({output.size(1)}, torch::TensorOptions().dtype(torch::kFloat64));
	}

	outputSum += batchOut;
	targetSum += batchTarget;
	count += n;
}

torch::Tensor calibration::Accumulator::meanOutput() const
{
	return outputSum / count;
}

torch::Tensor calibration::Accumulator::meanTarget() const
{
	return targetSum / count;
}

bool calibration::apply(const std::shared_ptr<ann::Net>& net, const std::vector<Accumulator>& points)
{
	std::vector<const Accumulator*> used;
	for(const Accumulator& acc : points)
		if(!acc.empty())
			used.push_back(&acc);

	if(used.empty())
	{
		Log(Log::ERROR)<<"No calibration spectra were supplied";
		return false;
	}
	if(used.size() > 2)
	{
		Log(Log::WARN)<<"Only the first two calibration files are used, "
			<<used.size()-2<<" were ignored";
	}

	int64_t outputs = net->getOutputSize();
	torch::Tensor scalars = net->getOutputScalars();
	torch::Tensor biases = net->getOutputBiases();
	if(scalars.numel() == 0 || biases.numel() == 0)
	{
		scalars = torch::ones({outputs}, torch::TensorOptions().dtype(torch::kFloat32));
		biases = torch::zeros({outputs}, torch::TensorOptions().dtype(torch::kFloat32));
	}
	scalars = scalars.to(torch::kFloat64).to(torch::kCPU);
	biases = biases.to(torch::kFloat64).to(torch::kCPU);

	torch::Tensor v1 = used[0]->meanOutput();
	torch::Tensor t1 = used[0]->meanTarget();

	torch::Tensor scale = torch::ones({outputs}, torch::TensorOptions().dtype(torch::kFloat64));
	torch::Tensor shift = t1 - v1;

	if(used.size() == 2)
	{
		torch::Tensor v2 = used[1]->meanOutput();
		torch::Tensor t2 = used[1]->meanTarget();
		torch::Tensor dv = v2 - v1;
		for(int64_t i = 0; i < outputs; ++i)
		{
			double d = dv[i].item<double>();
			if(std::fabs(d) < 1e-9)
			{
				Log(Log::WARN)<<"The two calibration files produce the same mean output for "
					<<"output "<<i<<", calibrating its bias from the first one only";
				continue;
			}
			double a = (t2[i].item<double>() - t1[i].item<double>()) / d;
			scale[i] = a;
			shift[i] = t1[i].item<double>() - a * v1[i].item<double>();
		}
	}

	net->setOutputScalars((scalars*scale).to(torch::kFloat32));
	net->setOutputBiases((biases + scalars*shift).to(torch::kFloat32));

	report(net, scale, shift);
	return true;
}

void calibration::report(const std::shared_ptr<ann::Net>& net, const torch::Tensor& scale,
		const torch::Tensor& bias)
{
	const std::vector<std::string>& labels = net->getOutputLabels();
	Log(Log::INFO)<<"Calibrated the network output:";
	for(int64_t i = 0; i < scale.numel(); ++i)
	{
		std::string name = (i < static_cast<int64_t>(labels.size())) ? labels[i] : std::to_string(i);
		Log(Log::INFO)<<"  "<<name<<(name.size() >= 8 ? "\t" : "\t\t")
			<<"scale "<<scale[i].item<double>()<<" bias "<<bias[i].item<double>();
	}
}
