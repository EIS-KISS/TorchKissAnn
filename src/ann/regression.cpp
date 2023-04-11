#include "regression.h"

using namespace ann::regression;

torch::Tensor ann::regression::use(torch::Tensor input, std::shared_ptr<Net> net)
{
	net->eval();
	torch::Tensor output = net->forward(input.reshape({1, net->getInputSize()}));
	return output;
}

