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
