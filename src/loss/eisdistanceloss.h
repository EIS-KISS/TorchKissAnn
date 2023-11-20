#pragma once

#include <eisgenerator/model.h>
#include <unistd.h>
#include <memory>

#include "torchph.h"

class EisDistanceLoss : public  torch::nn::Module
{
	std::shared_ptr<eis::Model> model;
	torch::Tensor omegas;
	std::shared_ptr<torch::CompilationUnit> modelScript;
	torch::nn::MSELoss loss;
	torch::Tensor targetScalar;

	public:
		EisDistanceLoss(std::string modelString, torch::Tensor omegas, torch::Tensor targetScalar = torch::Tensor());
		EisDistanceLoss(const eis::Model& model, torch::Tensor omegas, torch::Tensor targetScalar = torch::Tensor());

		torch::Tensor forward(torch::Tensor output, torch::Tensor targets);
		torch::Tensor distance(torch::Tensor output, torch::Tensor targetSpectra);
		inline torch::Tensor operator()(torch::Tensor output, torch::Tensor targets)
		{
			return forward(output, targets);
		}
};
