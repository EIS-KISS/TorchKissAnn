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

#include "permutationmsseloss.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <tuple>

#include "fit/fit.h"
#include "log.h"
#include "tensoroperators.h"

PermutationMseLoss::PermutationMseLoss(std::string modelStringIn, torch::Tensor omegasIn, torch::Tensor parameterScalarIn,
		torch::Tensor parameterBiasIn, torch::Tensor parameterLogIn, double toleranceIn, size_t limitIn):
modelString(modelStringIn),
parameterScalar(parameterScalarIn),
parameterBias(parameterBiasIn),
parameterLog(parameterLogIn),
tolerance(toleranceIn),
limit(limitIn)
{
	torch::Tensor omegas = omegasIn.detach().reshape({omegasIn.numel()}).to(torch::kFloat64).cpu();
	assert(omegas.numel() >= 2);
	omegaRange = eis::Range(omegas.min().item<double>(), omegas.max().item<double>(), omegas.numel(), true);

	eis::Model model(modelString);
	parameterCount = model.getParameterCount();
	Log(Log::INFO)<<"Permutation mse loss on "<<modelString<<" with "<<parameterCount
		<<" parameters, permutations probed from "<<omegaRange.start<<" to "<<omegaRange.end
		<<" rad/s, error measured in "<<(parameterScalar.numel() != 0 ? "the normalized parameter space" : "the space the network works in");
}

const std::vector<std::vector<double>>& PermutationMseLoss::candidatesFor(const std::vector<double>& parameters)
{
	std::map<std::vector<double>, std::vector<std::vector<double>>>::iterator cached = cache.find(parameters);
	if(cached != cache.end())
		return cached->second;

	std::vector<std::vector<double>> candidates;
	candidates.push_back(parameters);

	std::string withParameters = fittedModelString(modelString, torch::tensor(parameters).to(torch::kFloat64));
	if(withParameters.empty())
		Log(Log::DEBUG)<<"Can not embed the target parameters into "<<modelString<<", scoring against the target alone";
	else
	{
		try
		{
			eis::Model model(withParameters);
			std::vector<eis::ParameterPermutation> permutations =
				model.getEquivalentParameterPermutations(omegaRange, tolerance, limit);
			for(const eis::ParameterPermutation& permutation : permutations)
			{
				if(permutation.parameters.size() != parameters.size())
					continue;
				std::vector<double> values(parameters.size());
				for(size_t i = 0; i < values.size(); ++i)
					values[i] = permutation.parameters[i];
				candidates.push_back(std::move(values));
			}
			if(!permutations.empty())
				Log(Log::DEBUG)<<"Found "<<permutations.size()<<" equivalent permutation(s) of "<<withParameters;
		}
		catch(const std::exception& ex)
		{
			Log(Log::DEBUG)<<"Permutation analysis failed for "<<withParameters<<": "<<ex.what();
		}
	}

	return cache.emplace(parameters, std::move(candidates)).first->second;
}

torch::Tensor PermutationMseLoss::forward(torch::Tensor output, torch::Tensor targets)
{
	assert(targets.sizes() == output.sizes());

	const torch::Device computeDevice = output.device();
	const torch::ScalarType computeType = output.scalar_type();

	torch::Tensor predictions = output.to(torch::kCPU).to(torch::kFloat64);
	if(predictions.dim() > 1 && predictions.size(1) != parameterCount && predictions.size(0) == parameterCount)
		predictions = predictions.t();

	if(predictions.numel() % parameterCount != 0)
	{
		Log(Log::ERROR)<<"The network produced "<<predictions.numel()<<" values which is not a multiple of the "
			<<parameterCount<<" parameters of "<<modelString;
		throw std::runtime_error("Parameter count missmatch");
	}
	predictions = predictions.reshape({-1, parameterCount});

	torch::Tensor targetValues = targets.detach().to(torch::kCPU).to(torch::kFloat64);
	if(targetValues.dim() > 1 && targetValues.size(1) != parameterCount && targetValues.size(0) == parameterCount)
		targetValues = targetValues.t();
	if(parameterScalar.numel() != 0)
		targetValues = affineToPhysical(targetValues, parameterScalar, parameterBias, parameterLog);
	targetValues = targetValues.reshape({-1, parameterCount}).contiguous();
	assert(targetValues.size(0) == predictions.size(0));

	const int64_t batch = predictions.size(0);
	std::vector<double> raw(targetValues.size(0)*parameterCount);
	std::memcpy(raw.data(), targetValues.data_ptr<double>(), raw.size()*sizeof(double));

	std::vector<const std::vector<std::vector<double>>*> sets(batch);
	size_t candidateCount = 1;
	for(int64_t row = 0; row < batch; ++row)
	{
		const std::vector<double> target(raw.begin()+row*parameterCount, raw.begin()+(row+1)*parameterCount);
		sets[row] = &candidatesFor(target);
		candidateCount = std::max(candidateCount, sets[row]->size());
	}

	std::vector<double> buffer(candidateCount*batch*parameterCount);
	for(size_t candidate = 0; candidate < candidateCount; ++candidate)
	{
		for(int64_t row = 0; row < batch; ++row)
		{
			const std::vector<std::vector<double>>& set = *sets[row];
			const std::vector<double>& values = set[std::min(candidate, set.size()-1)];
			std::memcpy(&buffer[(candidate*batch+row)*parameterCount], values.data(), parameterCount*sizeof(double));
		}
	}

	torch::Tensor candidates = torch::from_blob(buffer.data(), {static_cast<int64_t>(candidateCount), batch, parameterCount},
			torch::TensorOptions().dtype(torch::kFloat64)).clone();

	if(parameterScalar.numel() != 0)
		candidates = physicalToAffine(candidates, parameterScalar, parameterBias, parameterLog);

	torch::Tensor difference = predictions.unsqueeze(0) - candidates;
	torch::Tensor msePerCandidate = torch::mean(difference*difference, -1);
	torch::Tensor best = std::get<0>(torch::min(msePerCandidate, 0));

	return torch::mean(best).to(computeDevice).to(computeType);
}
