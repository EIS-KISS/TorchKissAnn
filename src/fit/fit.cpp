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

#include "fit.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <eisgenerator/componant/componant.h>
#include <eisgenerator/model.h>

#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/jit/api/function_impl.h>

#include "modelscript.h"

namespace
{

constexpr double FIXED_WIDENING_DECADES = 3;
constexpr double SWEEP_WIDENING_DECADES = 6;
constexpr double TINY_PARAMETER = 1e-300;
constexpr double BOUND_SLACK = 1e-3;
constexpr double NOISE_AGREEMENT = 2.0;
constexpr double POOR_FIT_FACTOR = 5;
constexpr double CORRELATED_RESIDUAL = 0.8;

torch::Tensor doubles(const std::vector<double>& values)
{
	return torch::tensor(values).to(torch::kFloat64);
}

std::string formatValue(double value)
{
	std::ostringstream stream;
	stream<<std::setprecision(10)<<value;
	return stream.str();
}

struct Problem
{
	eis::Model model;
	std::string modelString;
	std::shared_ptr<torch::CompilationUnit> script;
	std::string functionName;
	torch::Tensor omegas;
	torch::Tensor target;
	torch::Tensor scale;
	torch::Tensor logLower;
	torch::Tensor logUpper;
	int64_t parameterCount = 0;
	int64_t evaluations = 0;

	explicit Problem(const std::string& from): model(from), modelString(from)
	{
		script = compileModel(model);
		functionName = model.getCompiledFunctionName();
	}

	torch::Tensor simulate(const torch::Tensor& parameters)
	{
		++evaluations;
		return runScriptModel(model, script, parameters, omegas);
	}

	 // The profiling executor caches its plan on the first call and reuses it whatever later inputs
	 // require. A plan built from a non-differentiable call carries no fused DifferentiableGraph, so
	 // autograd traces the primitive ops instead of the backward that hands the engine a complex
	 // gradient for the real parameter input and throws.
	void warmUp(const std::vector<double>& parameters)
	{
		simulate(doubles(parameters));
	}

	torch::Tensor clampToBox(const torch::Tensor& logParameters) const
	{
		return torch::clamp(logParameters, logLower, logUpper);
	}

	torch::Tensor residualOfPhysical(const torch::Tensor& parameters)
	{
		torch::Tensor difference = simulate(parameters) - target;
		torch::Tensor stacked = torch::stack({torch::real(difference), torch::imag(difference)}, 0);
		return (scale * stacked).reshape({stacked.numel()});
	}

	// The same residual as a function of the log10 parameters the optimiser moves around.
	torch::Tensor residual(const torch::Tensor& logParameters)
	{
		return residualOfPhysical(torch::pow(10.0, clampToBox(logParameters)));
	}

	double cost(const torch::Tensor& logParameters)
	{
		torch::Tensor r = residual(logParameters);
		return torch::sum(r*r).item<double>();
	}

	torch::Tensor jacobian(const torch::Tensor& logParameters)
	{
		torch::Tensor physical = torch::pow(10.0, clampToBox(logParameters));
		torch::Tensor leaf = physical.clone().detach().requires_grad_();
		torch::Tensor r = residualOfPhysical(leaf);

		std::vector<torch::Tensor> rows;
		rows.reserve(r.numel());
		for(int64_t k = 0; k < r.numel(); ++k)
		{
			torch::Tensor seed = torch::zeros_like(r);
			seed[k] = 1.0;
			torch::autograd::variable_list gradient = torch::autograd::grad({r}, {leaf}, {seed},
					true, false, true);
			rows.push_back(gradient.at(0).defined() ? gradient.at(0) : torch::zeros_like(leaf));
		}
		return torch::stack(rows, 0) * (physical * std::log(10.0)).reshape({1, -1});
	}
};

struct StartOutcome
{
	bool valid = false;
	bool converged = false;
	double value = std::numeric_limits<double>::quiet_NaN();
	torch::Tensor logParameters;
	int64_t evaluations = 0;
};

// Levenberg Marquardt on the log parameters.
StartOutcome runStart(Problem& problem, const torch::Tensor& startLogParameters, double tolerance,
		int64_t budget)
{
	StartOutcome outcome;
	torch::Tensor x = problem.clampToBox(startLogParameters).detach().clone();

	double cost = problem.cost(x);
	if(not std::isfinite(cost))
		return outcome;

	double lambda = 1e-2;
	const int64_t usedByStart = problem.evaluations;

	while(problem.evaluations - usedByStart < budget)
	{
		torch::Tensor j = problem.jacobian(x);
		torch::Tensor r = problem.residual(x);
		torch::Tensor normal = torch::t(j).matmul(j);
		torch::Tensor gradient = torch::t(j).matmul(r.reshape({-1, 1}));
		torch::Tensor curvature = torch::diagonal(normal);
		double scaleOfCurvature = torch::max(torch::abs(curvature)).item<double>();
		if(not std::isfinite(scaleOfCurvature) or scaleOfCurvature <= 0)
			break;

		bool improved = false;
		for(int attempt = 0; attempt < 30 and not improved; ++attempt)
		{
			torch::Tensor damped = normal + lambda*torch::diag(curvature)
				+ 1e-12*scaleOfCurvature*torch::eye(normal.size(0), normal.options());
			torch::Tensor step;
			try
			{
				step = torch::linalg_solve(damped, gradient).reshape({-1});
			}
			catch(const std::exception&)
			{
				lambda *= 10;
				continue;
			}
			if(not std::isfinite(torch::sum(step*step).item<double>()))
			{
				lambda *= 10;
				continue;
			}

			torch::Tensor candidate = problem.clampToBox(x - step);
			double candidateCost = problem.cost(candidate);
			// the tolerance gates what counts as an improvement at all, relative to the cost it is
			// measured against; testing the size of a decrease instead would call a large cost with a
			// tiny step converged
			if(std::isfinite(candidateCost) and candidateCost < cost*(1.0-tolerance))
			{
				cost = candidateCost;
				x = candidate;
				lambda = std::max(lambda*0.3, 1e-14);
				improved = true;
			}
			else
				lambda *= 10;
		}

		// nothing improved the objective however far the trust region was shrunk: a local minimum,
		// which is what Marquardt terminates on. Only running out of evaluations is not convergence.
		if(not improved)
		{
			outcome.converged = true;
			break;
		}
	}

	outcome.valid = std::isfinite(cost);
	outcome.value = cost;
	outcome.logParameters = x;
	outcome.evaluations = problem.evaluations - usedByStart;
	return outcome;
}

/**
 * Which parameters the model string actually carries a brace for.
 *
 * An element written bare gets the sweep range the generator substitutes for it, which is a range
 * chosen to make example spectra, not a bound anyone asked for. Only an element written with a brace
 * states an intent the fit should be held to, so the walk over the string that fittedModelString does
 * is done here too and one flag per parameter is returned.
 */
std::vector<bool> explicitParameters(const std::string& modelString,
		const std::vector<eis::Componant*>& componants)
{
	std::vector<bool> given;
	size_t element = 0;
	for(size_t i = 0; i < modelString.size() and element < componants.size(); ++i)
	{
		if(not std::isalpha(static_cast<unsigned char>(modelString[i])))
			continue;
		bool brace = i+1 < modelString.size() and modelString[i+1] == '{';
		for(size_t k = 0; k < componants[element]->paramCount(); ++k)
			given.push_back(brace);
		++element;
		if(brace)
			while(i < modelString.size() and modelString[i] != '}')
				++i;
	}
	return given;
}


// Decide the box and the default start of every parameter from the model string.
//
// Three cases. A value in braces is a starting guess and gets decades of freedom around it, so a
// circuit given as an estimate can still be corrected. A range in braces is a bound the caller asked
// for and is honoured exactly. A bare element carries whatever sweep range the generator substitutes,
// and that is widened, because a spectrum whose true resistance is 100 kOhm must not be reported as
// pinned to the 10 kOhm top of an example range.
void deriveBox(Problem& problem, double widenDecades, std::vector<double>& lower,
		std::vector<double>& upper, std::vector<double>& start)
{
	std::vector<eis::Range> ranges = problem.model.getFlatParameterRanges();
	std::vector<eis::Range> defaults = problem.model.getDefaultParameters();
	std::vector<bool> given = explicitParameters(problem.modelString,
			problem.model.getFlatComponants());
	problem.parameterCount = static_cast<int64_t>(ranges.size());

	for(size_t i = 0; i < ranges.size(); ++i)
	{
		const eis::Range& range = ranges[i];
		const eis::Range& fallback = i < defaults.size() ? defaults[i] : range;
		bool stated = i < given.size() and given[i];
		bool fixed = range.count <= 1 or range.start == range.end;
		bool logarithmic = fallback.log;

		double low = 0;
		double high = 0;
		double middle = 0;

		if(fixed)
		{
			double value = static_cast<double>(range.start);
			if(value <= 0)
				value = static_cast<double>(fallback.start);
			if(logarithmic)
			{
				double factor = std::pow(10.0, FIXED_WIDENING_DECADES);
				low = value/factor;
				high = value*factor;
			}
			else
			{
				double span = std::max(std::abs(value)*0.5, 1e-12);
				low = std::max(value-span, TINY_PARAMETER);
				high = value+span;
			}
			middle = value;
		}
		else if(stated)
		{
			low = static_cast<double>(range.start);
			high = static_cast<double>(range.end);
			middle = logarithmic ? std::sqrt(low*high) : 0.5*(low+high);
		}
		else
		{
			low = static_cast<double>(range.start);
			high = static_cast<double>(range.end);
			middle = logarithmic ? std::sqrt(low*high) : 0.5*(low+high);
			if(logarithmic)
			{
				double factor = std::pow(10.0, widenDecades);
				low = std::max(low/factor, TINY_PARAMETER);
				high = high*factor;
			}
			else
			{
				double slack = 0.5*(high-low)*widenDecades/SWEEP_WIDENING_DECADES;
				low = std::max(low-slack, TINY_PARAMETER);
				high = high+slack;
			}
		}

		// every parameter of a circuit is positive and a logarithm has to be finite
		low = std::max(low, TINY_PARAMETER);
		if(high <= low)
			high = low*10.0;

		// the start is pulled just inside the box, in log space, so that it is never taken from a
		// wall; a CPE exponent for instance is bounded by 1 and its geometric middle is fine
		double logLow = std::log10(low);
		double logHigh = std::log10(high);
		double logSpan = logHigh-logLow;
		double logMiddle = middle > 0 ? std::log10(middle) : 0.5*(logLow+logHigh);
		logMiddle = std::clamp(logMiddle, logLow+1e-3*logSpan, logHigh-1e-3*logSpan);
		middle = std::pow(10.0, logMiddle);

		lower.push_back(low);
		upper.push_back(high);
		start.push_back(middle);
	}
}


double estimateNoiseFloor(const torch::Tensor& target, const torch::Tensor& magnitude)
{
	if(target.numel() < 9)
		return 0;

	auto spread = [&target](const std::vector<double>& coefficients, double norm) {
		torch::Tensor acc;
		for(size_t shift = 0; shift < coefficients.size(); ++shift)
		{
			torch::Tensor slice = target.slice(0, shift, target.numel()-coefficients.size()+1+shift);
			torch::Tensor term = coefficients[shift]*slice;
			acc = acc.defined() ? acc + term : term;
		}
		return std::sqrt(torch::mean(torch::abs(acc)*torch::abs(acc)).item<double>())/norm;
	};

	double fourth = spread({1, -4, 6, -4, 1}, std::sqrt(70.0));
	double sixth = spread({1, -6, 15, -20, 15, -6, 1}, std::sqrt(924.0));

	if(not std::isfinite(fourth) or not std::isfinite(sixth))
		return 0;
	if(fourth > NOISE_AGREEMENT*sixth or sixth > NOISE_AGREEMENT*fourth)
		return 0;

	double mean = magnitude.mean().item<double>();
	if(not std::isfinite(mean) or mean <= 0)
		return 0;
	return 0.5*(fourth+sixth)/mean;
}

// Lag 1 autocorrelation of a sequence, close to one means it varies smoothly.
double lagOneAutocorrelation(const torch::Tensor& sequence)
{
	if(sequence.numel() < 3)
		return 0;
	torch::Tensor centred = sequence - sequence.mean();
	double energy = torch::sum(centred*centred).item<double>();
	if(not std::isfinite(energy) or energy <= 0)
		return 0;
	torch::Tensor first = centred.slice(0, 0, centred.numel()-1);
	torch::Tensor second = centred.slice(0, 1, centred.numel());
	return torch::sum(first*second).item<double>()/energy;
}

void applyVerdict(FitResult& result, double fitLimit)
{
	result.goodFitLimit = std::max(fitLimit, 3*result.noiseFloor);
	result.poorFitLimit = std::max(POOR_FIT_FACTOR*fitLimit, 10*result.noiseFloor);

	if(result.rmsRelativeError > result.poorFitLimit)
		result.verdict = FitVerdict::FIT_VERDICT_UNFITTED;
	else if(result.rmsRelativeError > result.goodFitLimit)
		result.verdict = FitVerdict::FIT_VERDICT_MARGINAL;
	else if(result.residualCorrelation > CORRELATED_RESIDUAL)
		result.verdict = FitVerdict::FIT_VERDICT_MARGINAL;
	else
		result.verdict = FitVerdict::FIT_VERDICT_FIT;
}

}

FitResult eisFit(torch::Tensor spectra, torch::Tensor omegas, const std::string& modelString,
		const FitOptions& options)
{
	FitResult result;

	Problem problem(modelString);

	problem.omegas = omegas.to(torch::kFloat64).reshape({omegas.numel(), 1});
	problem.target = spectra.to(torch::kComplexDouble).reshape({spectra.numel()});
	if(problem.target.numel() != problem.omegas.numel() or problem.target.numel() < 2)
		return result;

	torch::Tensor magnitude = torch::abs(problem.target);
	torch::Tensor floor = torch::mean(magnitude)*options.weightFloor;
	torch::Tensor weight = 1.0/torch::sqrt(magnitude*magnitude + floor*floor);
	problem.scale = torch::sqrt(weight/problem.target.numel());

	std::vector<double> lower;
	std::vector<double> upper;
	std::vector<double> start;
	deriveBox(problem, 0, lower, upper, start);
	// the narrow box is kept for the restarts to sample even after the box is widened: a point drawn
	// from twelve decades is almost always hopeless, one drawn from the element's own range usually
	// lands in a real basin
	std::vector<double> narrowLower = lower;
	std::vector<double> narrowUpper = upper;

	if(options.startingParams.defined() and options.startingParams.numel() == problem.parameterCount)
	{
		torch::Tensor given = options.startingParams.to(torch::kFloat64).reshape({problem.parameterCount});
		auto accessor = given.accessor<double, 1>();
		for(int64_t i = 0; i < problem.parameterCount; ++i)
			if(std::isfinite(accessor[i]) and accessor[i] > 0)
				start[i] = std::clamp(accessor[i], lower[i]*10.0, upper[i]*0.1);
	}

	problem.warmUp(start);

	result.parameterNames = problem.model.getParameterNames();
	result.uncertainties = torch::full({problem.parameterCount}, std::numeric_limits<double>::quiet_NaN(),
			torch::kFloat64);

	std::mt19937_64 generator(options.seed);
	std::uniform_real_distribution<double> uniform(0.02, 0.98);
	std::uniform_int_distribution<int64_t> pick(0, std::max<int64_t>(0, problem.parameterCount-1));

	double bestValue = std::numeric_limits<double>::infinity();
	StartOutcome best;
	std::vector<double> bestStart = start;

	for(int stage = 0; stage < 2; ++stage)
	{
		std::vector<double> stageLower;
		std::vector<double> stageUpper;
		std::vector<double> stageStart;
		deriveBox(problem, stage == 0 ? 0 : SWEEP_WIDENING_DECADES, stageLower, stageUpper, stageStart);

		if(stage == 1)
		{
			// seed the wide stage from what the narrow one actually found, not from where it began,
			// and keep the wide box for the report
			torch::Tensor settled = torch::pow(10.0, problem.clampToBox(best.logParameters));
			auto accessor = settled.accessor<double, 1>();
			for(int64_t i = 0; i < problem.parameterCount; ++i)
				stageStart[i] = std::clamp(accessor[i], stageLower[i], stageUpper[i]);
			lower = stageLower;
			upper = stageUpper;
		}

		problem.logLower = doubles(stageLower).log10();
		problem.logUpper = doubles(stageUpper).log10();

		std::vector<double> home;
		for(double value: stageStart)
			home.push_back(std::log10(value));

		for(int attempt = 0; attempt <= options.restarts; ++attempt)
		{
			std::vector<double> candidate;
			if(attempt == 0)
				candidate = home;
			else if(best.valid and attempt % 2 == 1)
			{
				// basin hopping by coordinate: keep everything the last start settled on and redraw
				// one parameter anywhere in its box. The element that has to become a short or an open
				// circuit is a single parameter and it can be many decades away, which a small nudge
				// of all of them never reaches.
				candidate.clear();
				int64_t which = pick(generator);
				for(int64_t i = 0; i < problem.parameterCount; ++i)
				{
					if(i == which)
						candidate.push_back(std::log10(stageLower[i])
							+ uniform(generator)*(std::log10(stageUpper[i])-std::log10(stageLower[i])));
					else
						candidate.push_back(best.logParameters[i].item<double>());
				}
			}
			else
			{
				candidate = home;
				for(int64_t i = 0; i < problem.parameterCount; ++i)
					candidate[i] = std::log10(narrowLower[i])
						+ uniform(generator)*(std::log10(narrowUpper[i])-std::log10(narrowLower[i]));
			}

			StartOutcome outcome = runStart(problem, doubles(candidate), options.tolerance,
					std::max<int64_t>(1, options.maxEvaluations));
			if(not outcome.valid)
				continue;

			++result.starts;
			if(outcome.value < bestValue)
			{
				bestValue = outcome.value;
				best = outcome;
				bestStart = stageStart;
				result.bestStart = result.starts;
			}
		}
	}

	if(not best.valid)
		return result;

	torch::NoGradGuard noGrad;
	result.parameters = torch::pow(10.0, problem.clampToBox(best.logParameters)).detach();
	result.success = true;
	result.converged = best.converged;
	result.startParameters = doubles(bestStart);
	result.lowerBounds = doubles(lower);
	result.upperBounds = doubles(upper);
	result.evaluations = best.evaluations;
	result.chiSquared = best.value;

	const int64_t points = problem.target.numel();
	const int64_t dof = std::max<int64_t>(1, 2*points - problem.parameterCount);
	result.reducedChiSquared = best.value*2.0*points/dof;

	result.fittedSpectra = problem.simulate(result.parameters);
	result.residuals = problem.target - result.fittedSpectra;
	torch::Tensor error = torch::abs(result.residuals)/magnitude;
	result.rmsRelativeError = torch::mean(error*error).item<double>();
	result.rmsRelativeError = std::sqrt(result.rmsRelativeError);
	result.maxRelativeError = error.max().item<double>();
	result.noiseFloor = estimateNoiseFloor(problem.target, magnitude);
	result.residualCorrelation = lagOneAutocorrelation(error);

	for(int64_t i = 0; i < problem.parameterCount; ++i)
	{
		double logValue = best.logParameters[i].item<double>();
		double logSpan = problem.logUpper[i].item<double>()-problem.logLower[i].item<double>();
		result.atBound.push_back(logValue-problem.logLower[i].item<double>() <= BOUND_SLACK*logSpan
				or problem.logUpper[i].item<double>()-logValue <= BOUND_SLACK*logSpan);
	}

	// the curvature of the residual around the optimum gives one sigma per parameter and says how
	// far the parameters can be traded against each other without changing the spectra
	try
	{
		torch::AutoGradMode enableGrad(true);
		torch::Tensor j = problem.jacobian(best.logParameters);
		torch::Tensor singular = torch::linalg_svdvals(j);
		double largest = singular[0].item<double>();
		double smallest = singular[singular.numel()-1].item<double>();
		result.conditionNumber = smallest > 0 ? largest/smallest : std::numeric_limits<double>::infinity();

		torch::Tensor r = problem.residual(best.logParameters);
		double variance = torch::sum(r*r).item<double>()/dof;
		torch::Tensor diagonal = torch::diagonal(torch::linalg_inv(torch::t(j).matmul(j))*variance);
		for(int64_t i = 0; i < problem.parameterCount; ++i)
		{
			double entry = diagonal[i].item<double>();
			result.uncertainties[i] = std::isfinite(entry) and entry > 0 ? std::sqrt(entry)
				: std::numeric_limits<double>::quiet_NaN();
		}
		// the Jacobian was taken in log10 space, so the sigma is a fraction of the parameter
		result.uncertainties = result.uncertainties*result.parameters*std::log(10.0);
	}
	catch(const std::exception&)
	{
		// a singular normal matrix leaves the uncertainties at NaN
	}

	applyVerdict(result, options.fitLimit);

	return result;
}

std::string fittedModelString(const std::string& modelString, const torch::Tensor& parameters)
{
	if(not parameters.defined())
		return "";

	eis::Model model(modelString);
	std::vector<eis::Componant*> componants = model.getFlatComponants();
	torch::Tensor values = parameters.to(torch::kFloat64).reshape({parameters.numel()});
	auto accessor = values.accessor<double, 1>();

	std::string out;
	size_t componantIndex = 0;
	int64_t valueIndex = 0;

	for(size_t i = 0; i < modelString.size(); ++i)
	{
		char symbol = modelString[i];
		if(not std::isalpha(static_cast<unsigned char>(symbol)))
		{
			out += symbol;
			continue;
		}

		if(componantIndex >= componants.size())
			return "";
		size_t count = componants[componantIndex]->paramCount();
		++componantIndex;
		if(valueIndex+static_cast<int64_t>(count) > values.numel())
			return "";

		out += symbol;
		out += '{';
		for(size_t parameter = 0; parameter < count; ++parameter)
		{
			if(parameter > 0)
				out += ", ";
			out += formatValue(accessor[valueIndex++]);
		}
		out += '}';

		// drop the range the model string carried for this element
		if(i+1 < modelString.size() and modelString[i+1] == '{')
			while(i < modelString.size() and modelString[i] != '}')
				++i;
	}

	if(componantIndex != componants.size() or valueIndex != values.numel())
		return "";
	return out;
}
