#include "net.h"

Net::Net(int64_t inputSizeI, int64_t outputSizeI, bool leakI): inputSize(inputSizeI), outputSize(outputSizeI), leak(leakI)
{
	// Construct and register two Linear submodules.
	fc1 = register_module("fc1", torch::nn::Linear(inputSize, inputSize/2));
	fc2 = register_module("fc2", torch::nn::Linear(inputSize/2, inputSize/4));
	fc3 = register_module("fc3", torch::nn::Linear(inputSize/4, outputSize*4));
	fc4 = register_module("fc4", torch::nn::Linear(outputSize*4, outputSize));
}

torch::Tensor Net::forward(torch::Tensor x)
{
	torch::nn::functional::LeakyReLUFuncOptions leakyOptions;
	leakyOptions = leakyOptions.negative_slope(0.42).inplace(true);

	x = torch::relu(fc1->forward(x));
	x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
	if(leak)
		x = torch::nn::functional::leaky_relu(fc2->forward(x), leakyOptions);
	else
		x = torch::relu(fc2->forward(x));
	x = torch::dropout(x, /*p=*/0.1, /*train=*/is_training());
	if(leak)
		x = torch::nn::functional::leaky_relu(fc3->forward(x), leakyOptions);
	else
		x = torch::relu(fc3->forward(x));
	x = torch::log_softmax(fc4->forward(x), 1);
	return x;
}

int64_t Net::getInputSize()
{
	return inputSize;
}

int64_t Net::getOutputSize()
{
	return outputSize;
}
