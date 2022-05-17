#include "net.h"
#include "eisgeneratorloader.h"

Net::Net()
{
	// Construct and register two Linear submodules.
	fc1 = register_module("fc1", torch::nn::Linear(INPUT_SIZE, 30));
	fc2 = register_module("fc2", torch::nn::Linear(30, 10));
	fc3 = register_module("fc3", torch::nn::Linear(10, OUTPUT_SIZE));
}

torch::Tensor Net::forward(torch::Tensor x)
{
	x = torch::relu(fc1->forward(x));
	//x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
	x = torch::relu(fc2->forward(x));
	x = torch::log_softmax(fc3->forward(x), 1);
	return x;
}

int64_t Net::getInputSize()
{
	return INPUT_SIZE;
}

int64_t Net::getOutputSize()
{
	return OUTPUT_SIZE;
}
