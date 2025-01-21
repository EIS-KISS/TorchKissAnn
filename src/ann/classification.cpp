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

#include "classification.h"

#include <kisstype/spectra.h>

#include "log.h"
#include "randomgen.h"
#include "../data/eistotorch.h"

using namespace ann::classification;

torch::Tensor ann::classification::multiClassHits(const torch::Tensor& prediction, torch::Tensor targets, double thresh)
{
	torch::Tensor hits = torch::ones({prediction.size(0)},
		torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
	for(int64_t i = 0; i < prediction.size(0); ++i)
	{
		for(int64_t j = 0; j < prediction.size(1); ++j)
		{
			if(((targets[i][j].template item<float>()) >= thresh && prediction[i][j].template item<float>() < thresh) ||
				((targets[i][j].template item<float>()) < thresh && prediction[i][j].template item<float>() >= thresh))
			{
				hits[i] = 0;
			}
		}
	}

	return hits;
}

torch::Tensor ann::classification::use(torch::Tensor input, std::shared_ptr<Net> net)
{
	net->eval();
	torch::Tensor output = torch::exp(net->forward(input.reshape({1, net->getInputSize()})));
	return output;
}


static const std::string getSampleOutputFileName(const std::string& dir, bool succ, int classNum)
{
	return dir + (succ ? std::string("/succ_") : std::string("/fail_")) +
			std::to_string(classNum) + std::string("_") + std::to_string(rd::uid()) + ".spc";
}

void ann::classification::exportRandomSamples(const torch::Tensor& data, const torch::Tensor& targets,
						 const torch::Tensor& hits, const std::filesystem::path& outDir,
						 bool outputProb)
{
	for(int i = 0; i < hits.size(0); ++i)
	{
		if(hits[i].template item<int>() > 0)
		{
			if(outputProb > 0 && rd::rand() < (outputProb/10.0f))
			{
				for(int64_t j = 0; j < targets[i].size(0); ++j)
				{
					if((targets[i][j].template item<int64_t>()) == 1)
						eis::Spectra(torchToEis(data.select(0, i)), "", "").saveToDisk(getSampleOutputFileName(outDir, true, j));
				}
			}
		}
		else if(outputProb > 0 && rd::rand() < outputProb)
		{
			for(int64_t j = 0; j < targets[i].size(0); ++j)
			{
				if((targets[i][j].template item<int64_t>()) == 1)
				{
					eis::Spectra spectra(torchToEis(data.select(0, i)), "", "");
					spectra.saveToDisk(getSampleOutputFileName(outDir, false, j));
				}
			}
		}
	}
}
