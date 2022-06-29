#include "simplenet.h"
#include <cstddef>
#include <utility>

static std::pair<size_t, size_t> layerWidthFunction(size_t layer, size_t leyers, size_t input, size_t output)
{
	size_t layerIn = (input*(leyers-layer)/leyers)+output*(layer)/leyers;
	size_t layerOut = (input*(leyers-layer-1)/leyers)+output*(layer+1)/leyers;
	return std::pair<size_t, size_t>(layerIn, layerOut);
}

GANGeneratorImpl::GANGeneratorImpl(size_t inputSize, size_t outputSize)
{
	register_module("GANGeneratorModel", model);
	std::pair<size_t, size_t> layerSizes = layerWidthFunction(0, 4, inputSize, outputSize);
	model->push_back(torch::nn::Linear(layerSizes.first, layerSizes.second));
	model->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)));
	layerSizes = layerWidthFunction(1, 4, inputSize, outputSize);
	model->push_back(torch::nn::Linear(layerSizes.first, layerSizes.second));
	model->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)));
	layerSizes = layerWidthFunction(2, 4, inputSize, outputSize);
	model->push_back(torch::nn::Linear(layerSizes.first, layerSizes.second));
	model->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)));
	layerSizes = layerWidthFunction(3, 4, inputSize, outputSize);
	model->push_back(torch::nn::Linear(layerSizes.first, layerSizes.second));
	model->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)));
	model->push_back(torch::nn::Tanh());
}

torch::Tensor GANGeneratorImpl::forward(torch::Tensor x)
{
	return model->forward(x);
}

GANDiscriminatorImpl::GANDiscriminatorImpl(size_t inputSize)
{
	register_module("GANDiscriminatorFeature", feature);
	register_module("GANDiscriminatorClassifier", classifier);
	std::pair<size_t, size_t> layerSizes = layerWidthFunction(0, 4, inputSize, 1);
	feature->push_back(torch::nn::Linear(layerSizes.first, layerSizes.second));
	feature->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)));
	layerSizes = layerWidthFunction(1, 4, inputSize, 1);
	feature->push_back(torch::nn::Linear(layerSizes.first, layerSizes.second));
	feature->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)));
	layerSizes = layerWidthFunction(2, 4, inputSize, 1);
	feature->push_back(torch::nn::Linear(layerSizes.first, layerSizes.second));
	feature->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)));
	layerSizes = layerWidthFunction(3, 4, inputSize, 1);
	classifier->push_back(torch::nn::Linear(layerSizes.first, layerSizes.second));
}

std::pair<torch::Tensor, torch::Tensor> GANDiscriminatorImpl::forward(torch::Tensor x)
{
	torch::Tensor features = feature->forward(x);
	torch::Tensor out = classifier->forward(features);
	return std::pair<torch::Tensor, torch::Tensor>(out, features);
}

