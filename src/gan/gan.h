#pragma once
#include "../data/generators/eisgendata.h"
#include "../utils/randomgen.h"
#include "../data/eistotorch.h"

namespace gan
{
	void train(std::string fileName = "ganNet", size_t exampleCount = EisGeneratorDataset::DEFAULT_EXAMPLE_COUNT, double noise = 0);
	bool filter(torch::Tensor input, const std::string& fileName, float nabla = 20);
	torch::Tensor generate(const std::string& fileName);
}
