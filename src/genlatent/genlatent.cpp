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

#include <cstdint>
#include <filesystem>
#include <ios>
#include <iostream>
#include <eisgenerator/model.h>
#include <eisgenerator/log.h>
#include <memory>
#include <fstream>
#include <torch/torch.h>

#include "ann/autoencoder.h"
#include "ann/autoencode.h"
#include "commonoptions.h"
#include "data/loaders/tarloader.h"
#include "log.h"
#include "data/loaders/dirloader.h"
#include "data/loaders/regressiondirloader.h"
#include "data/loaders/regressionloader.h"
#include "options.h"
#include "globals.h"
#include "save.h"
#include "colortabel.h"
#include "tensoroperators.h"

template <typename T>
int exportLatentSpace(std::shared_ptr<ann::AutoEncoder> net, T& dataset, const std::filesystem::path& out)
{
	net->eval();

	torch::data::DataLoaderOptions options;
	options = options.batch_size(batch_size).workers(JOBS);
	auto dataLoader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(dataset.map(torch::data::transforms::Stack<>()), options);
	torch::nn::MSELoss mseloss(torch::nn::MSELossOptions().reduction(torch::kMean));
	ann::autoencode::TestReturn ret = ann::autoencode::test(net, *dataLoader, mseloss, dataset.size().value());

	std::fstream latentFile(out, std::ios_base::out | std::ios_base::trunc);
	if(!latentFile.is_open())
	{
		Log(Log::ERROR)<<"Could not open "<<out;
		return 1;
	}

	latentFile<<std::scientific;
	latentFile<<"LATF, 1.0.0\n";
	latentFile<<net->getLatentSize()<<", "<<1<<", "<<dataset.outputSize()<<'\n';

	for(size_t i = 0; i < dataset.outputSize(); ++i)
		latentFile<<dataset.outputName(i)<<'\n';

	for(int64_t i = 0; i < net->getLatentSize(); ++i)
		latentFile<<"dim"<<i<<", ";
	latentFile<<"class\n";

	auto latentAcesssor = ret.latents.accessor<float, 2>();
	for(int64_t i = 0; i < ret.latents.size(0); ++i)
	{
		for(int64_t j = 0; j < net->getLatentSize(); ++j)
			latentFile<<latentAcesssor[i][j]<<", ";
		latentFile<<ret.datasetTargets[i][0].item().toInt()<<'\n';
	}

	return 0;
}

int main(int argc, char** argv)
{
	std::cout<<std::setprecision(5)<<std::fixed<<std::setw(3);
	Log::level = Log::INFO;
	eis::Log::level = eis::Log::ERROR;

	Config config;
	argp_parse(&argp, argc, argv, 0, 0, &config);

	choose_device(config.noGpu);

	if(config.netpath.empty())
	{
		Log(Log::ERROR)<<"A path to a checkpoint directory must be given";
		return 1;
	}

	if(config.fileName.empty())
	{
		Log(Log::ERROR)<<"A path to a dataset is required";
		return 1;
	}

	std::shared_ptr<ann::Net> net(ann::Net::newNetFromCheckpointDir(config.netpath));

	if(!net)
	{
		Log(Log::ERROR)<<"Could not load network from "<<config.netpath;
		return 1;
	}

	std::shared_ptr<ann::AutoEncoder> autoencoder = std::dynamic_pointer_cast<ann::AutoEncoder>(net);
	if(!autoencoder)
	{
		Log(Log::ERROR)<<"The supplied network is not an autoencoder";
		return 1;
	}

	int ret = 2;
	switch(config.datasetMode)
	{
		case DATASET_DIR:
		{
			EisDirDataset dataset(config.fileName);
			ret = exportLatentSpace(autoencoder, dataset, config.outputFilename);
			break;
		}
		case DATASET_TAR:
		{
			EisTarDataset dataset(config.fileName);
			ret = exportLatentSpace(autoencoder, dataset, config.outputFilename);
			break;
		}
		case DATASET_INVALID:
		default:
			Log(Log::ERROR)<<"Invalid dataset type";
	}

	Log(Log::ERROR)<<"";

	free_device();
	return ret;
}
