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

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <eisgenerator/log.h>
#include <eisgenerator/model.h>
#include <iomanip>
#include <iostream>
#include <kisstype/spectra.h>
#include <kisstype/type.h>
#include <sstream>
#include <string>
#include <torch/torch.h>
#include <vector>

#include "data/eistotorch.h"
#include "fit/fit.h"
#include "globals.h"
#include "log.h"
#include "options.h"

namespace
{

std::string sci(double value, int precision = 4)
{
	std::ostringstream stream;
	stream<<std::scientific<<std::setprecision(precision)<<value;
	return stream.str();
}

std::string percent(double fraction, int precision = 3)
{
	std::ostringstream stream;
	stream<<std::fixed<<std::setprecision(precision)<<fraction*100<<" %";
	return stream.str();
}

void printParameters(const FitResult& fit)
{
	std::cout<<"\nParameter           Start         Fitted        +/- 1 sigma   Relative    Note\n";

	for(int64_t i = 0; i < fit.parameters.numel(); ++i)
	{
		double value = fit.parameters[i].item<double>();
		double start = fit.startParameters[i].item<double>();
		double error = fit.uncertainties[i].item<double>();

		std::string note;
		if(fit.atBound[i])
			note = "on bound";
		else if(std::isnan(error))
			note = "not identifiable";

		std::cout<<"  "<<std::setw(2)<<i<<' '<<std::left<<std::setw(14)<<fit.parameterNames[i]
			<<std::right<<std::setw(11)<<sci(start)
			<<std::setw(12)<<sci(value)
			<<std::setw(14)<<sci(error)
			<<std::setw(12)<<percent(std::abs(error/value))
			<<' '<<note<<'\n';
	}
}

void printQuality(const FitResult& fit)
{
	std::cout<<"\nFit quality\n"
		<<"  chi^2 per point          "<<sci(fit.chiSquared)<<'\n'
		<<"  reduced chi^2            "<<sci(fit.reducedChiSquared)<<'\n'
		<<"  rms relative residual    "<<percent(fit.rmsRelativeError)<<'\n'
		<<"  largest point residual   "<<percent(fit.maxRelativeError)<<'\n'
		<<"  noise floor of data      "<<(fit.noiseFloor > 0 ? percent(fit.noiseFloor) : "not estimable")<<'\n'
		<<"  residual autocorrelation "<<std::fixed<<std::setprecision(2)<<fit.residualCorrelation<<'\n'
		<<"  condition number         "<<sci(fit.conditionNumber)<<'\n'
		<<"  optimiser                "<<(fit.converged ? "converged" : "out of budget")<<", "<<fit.evaluations
		<<" simulations, start "<<fit.bestStart<<" of "<<fit.starts<<'\n';
}

/**
 * Say why eisFit() reached the verdict it did; the limits themselves are chosen by the fit, see
 * FitResult::goodFitLimit.
 */
void printVerdict(const FitResult& fit)
{
	switch(fit.verdict)
	{
		case FitVerdict::DoesNotFit:
			Log(Log::INFO)<<"Verdict: the model does not fit the spectra, its residual is "
				<<percent(fit.rmsRelativeError)<<" where at worst "<<percent(fit.poorFitLimit)
				<<" should be reachable";
			break;
		case FitVerdict::Marginal:
			if(fit.rmsRelativeError > fit.goodFitLimit)
				Log(Log::INFO)<<"Verdict: marginal, the residual of "<<percent(fit.rmsRelativeError)
					<<" is above the "<<percent(fit.goodFitLimit)<<" a fitting model should reach";
			else
				Log(Log::INFO)<<"Verdict: marginal, the residual of "<<percent(fit.rmsRelativeError)
					<<" is small but varies smoothly with the frequency (autocorrelation "
					<<std::fixed<<std::setprecision(2)<<fit.residualCorrelation
					<<"), the model misses a feature of the spectra";
			break;
		case FitVerdict::Fits:
			Log(Log::INFO)<<"Verdict: the model fits the spectra, its residual of "
				<<percent(fit.rmsRelativeError)<<" is within the "<<percent(fit.goodFitLimit)<<" limit";
			break;
	}
}

void printWarnings(const FitResult& fit)
{
	if(!fit.converged)
		Log(Log::WARN)<<"the optimiser ran out of its budget, raise --evaluations if the residual looks too large";

	for(size_t i = 0; i < fit.atBound.size(); ++i)
		if(fit.atBound[i])
			Log(Log::WARN)<<fit.parameterNames[i]<<" ended on a bound of "
				<<sci(fit.lowerBounds[static_cast<int64_t>(i)].item<double>())<<" to "
				<<sci(fit.upperBounds[static_cast<int64_t>(i)].item<double>())
				<<", give it a wider range in the model string if that is not intended";

	if(fit.conditionNumber > 1e6 || std::isinf(fit.conditionNumber))
		Log(Log::WARN)<<"the parameters are degenerate (condition number "<<sci(fit.conditionNumber)
			<<"), the spectra do not determine them individually";
}

bool writeFittedSpectra(const Config& config, const FitResult& fit, const std::vector<eis::DataPoint>& measured,
		const std::string& modelString)
{
	if(modelString.empty())
	{
		Log(Log::WARN)<<"the fitted parameters do not fit back into the model string, writing the spectra without one";
	}

	torch::Tensor fitted = fit.fittedSpectra.to(torch::kComplexDouble);
	auto accessor = fitted.accessor<c10::complex<double>, 1>();

	std::vector<eis::DataPoint> points;
	points.reserve(measured.size());
	for(size_t i = 0; i < measured.size(); ++i)
		points.push_back(eis::DataPoint(std::complex<fvalue>(static_cast<fvalue>(accessor[i].real()),
				static_cast<fvalue>(accessor[i].imag())), measured[i].omega));

	try
	{
		eis::Spectra spectra(points, modelString, "Fitted to "+config.fileName.string()+" by torchkissann_fit, rms residual "
				+percent(fit.rmsRelativeError));
		spectra.saveToDisk(config.outputFilename);
	}
	catch(const eis::file_error& error)
	{
		Log(Log::ERROR)<<"could not write "<<config.outputFilename<<": "<<error.what();
		return false;
	}

	Log(Log::INFO)<<"Wrote the spectrum of the fitted circuit to "<<config.outputFilename;
	return true;
}

}

int main(int argc, char** argv)
{
	Log::level = Log::INFO;
	eis::Log::level = eis::Log::ERROR;

	Config config;
	argp_parse(&argp, argc, argv, 0, 0, &config);

	if(config.fileName.empty())
	{
		Log(Log::ERROR)<<"A spectrum to fit must be given with -f";
		return 1;
	}
	if(config.modelString.empty())
	{
		Log(Log::ERROR)<<"A model to fit must be given with -m";
		return 1;
	}
	if(!(config.fitLimit > 0))
	{
		Log(Log::ERROR)<<"--fit-limit must be a positive fraction";
		return 1;
	}

	// a single spectra and a handful of parameters: the simulation is far too small for a gpu to
	// be worth the transfer, the fit is always carried out on the cpu
	choose_device(true);

	std::vector<eis::DataPoint> measured;
	try
	{
		measured = eis::Spectra::loadFromDisk(config.fileName).data;
	}
	catch(const eis::file_error& error)
	{
		Log(Log::ERROR)<<"could not load "<<config.fileName<<": "<<error.what();
		return 1;
	}

	if(measured.size() < 3)
	{
		Log(Log::ERROR)<<config.fileName<<" contains "<<measured.size()<<" usable points, at least 3 are needed";
		return 1;
	}
	std::sort(measured.begin(), measured.end());

	torch::Tensor omegas;
	torch::Tensor target = eisToComplexTensor(measured, &omegas);

	FitOptions options;
	options.restarts = config.restarts;
	options.maxEvaluations = config.evaluations;
	options.seed = config.seed;
	options.fitLimit = config.fitLimit;

	Log(Log::INFO)<<"Fitting "<<config.modelString<<" to "<<config.fileName<<" ("<<measured.size()
		<<" points, "<<sci(measured.front().omega)<<" to "<<sci(measured.back().omega)<<" rad/s)";

	FitResult fit;
	try
	{
		fit = eisFit(target, omegas, config.modelString, options);
	}
	catch(const std::exception& error)
	{
		// eis::Model throws on a circuit it cannot parse or compile, which is a bad command line
		// rather than a failed fit
		Log(Log::ERROR)<<"could not use the model "<<config.modelString<<": "<<error.what();
		return 1;
	}
	if(!fit.success)
	{
		Log(Log::ERROR)<<"no fit could be produced for "<<config.modelString;
		return 2;
	}

	printParameters(fit);
	printQuality(fit);

	std::string modelString = fittedModelString(config.modelString, fit.parameters);
	if(!modelString.empty())
		Log(Log::INFO)<<"Fitted model: "<<modelString;

	printVerdict(fit);
	printWarnings(fit);

	if(!config.outputFilename.empty() && !writeFittedSpectra(config, fit, measured, modelString))
		return 3;

	return fit.verdict == FitVerdict::DoesNotFit ? 4 : 0;
}
