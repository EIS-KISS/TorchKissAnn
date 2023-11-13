#include "regression.h"

#include <cstdint>
#include <eisgenerator/model.h>
#include <torch/optim/adamw.h>

using namespace ann::regression;

torch::Tensor ann::regression::use(torch::Tensor input, std::shared_ptr<Net> net)
{
	torch::NoGradGuard noGrad;
	net->eval();
	torch::Tensor output = net->forward(input.reshape({1, net->getInputSize()}));
	torch::Tensor outputScalars = net->getOutputScalars();
	torch::Tensor outputBias    = net->getOutputBiases();
	Log(Log::DEBUG)<<"Output:\n"<<output<<"\nOutputScalers:\n"<<outputScalars;
	output = (output*outputScalars+outputBias).reshape({output.numel()});
	return output;
}

std::pair<torch::Tensor, torch::Tensor>  ann::regression::getTargetScalesAndBiases(const std::string& modelString)
{
	eis::Model model(modelString);
	std::vector<eis::Range> defaultRanges = model.getDefaultParameters();
	torch::Tensor scales = torch::empty({static_cast<int64_t>(defaultRanges.size())}, tensorOptCpu<fvalue>(false));
	torch::Tensor bias = torch::empty({static_cast<int64_t>(defaultRanges.size())}, tensorOptCpu<fvalue>(false));

	for(size_t i = 0; i < defaultRanges.size(); ++i)
	{
		scales[i] = defaultRanges[i].end-defaultRanges[i].start;
		bias[i] = defaultRanges[i].start;
	}
	return {scales, bias};
}

