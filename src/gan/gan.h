#pragma once
#include "../data/generators/eisgendata.h"
#include "../utils/randomgen.h"
#include "../data/eistotorch.h"

namespace gan
{
	void train(std::string fileName = "ganNet", int sweepDepth = EisGeneratorDataset::DEFAULT_SWEEP_DEPTH, double noise = 0);
	bool filter(torch::Tensor input, const std::string& fileName, float nabla = 20);
	torch::Tensor generate(const std::string& fileName);
}
