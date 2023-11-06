#include "regression.h"

#include <cstdint>
#include <eisgenerator/model.h>

using namespace ann::regression;

torch::Tensor ann::regression::use(torch::Tensor input, std::shared_ptr<Net> net)
{
	torch::NoGradGuard noGrad;
	net->eval();
	torch::Tensor output = net->forward(input.reshape({1, net->getInputSize()}));
	output = output.reshape({output.numel()});
	return output;
}

torch::Tensor ann::regression::getTargetScalar(const std::string& modelString)
{
	eis::Model model(modelString);
	std::vector<eis::Range> defaultRanges = model.getFlatParameterRanges();
	torch::Tensor out = torch::empty({static_cast<int64_t>(defaultRanges.size())}, tensorOptCpu<fvalue>(false));

	for(size_t i = 0; i < defaultRanges.size(); ++i)
		out[i] = defaultRanges[i].center();
	return out;
}

