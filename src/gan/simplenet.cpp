//
// TorchKissAnn - A collection of tools to train various types of Machine learning
// algorithms on various types of EIS data
// Copyright (C) 2025 Carl Klemm <carl@uvos.xyz>
//
// This file is part of TorchKissAnn.
//
// TorchKissAnn is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// TorchKissAnn is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with TorchKissAnn.  If not, see <http://www.gnu.org/licenses/>.
//

#include "simplenet.h"
#include <cstddef>
#include <utility>
#include <fstream>
#include <json/value.h>

static std::pair<size_t, size_t> layerWidthFunction(size_t layer, size_t leyers, size_t input, size_t output)
{
	size_t layerIn = (input*(leyers-layer)/leyers)+output*(layer)/leyers;
	size_t layerOut = (input*(leyers-layer-1)/leyers)+output*(layer+1)/leyers;
	return std::pair<size_t, size_t>(layerIn, layerOut);
}

GANGeneratorImpl::GANGeneratorImpl(const Json::Value& node):
Net(node)
{
	init();
}

GANGeneratorImpl::GANGeneratorImpl(size_t inputSize, size_t outputSize):
Net(inputSize, outputSize, false)
{
	init();
}

void GANGeneratorImpl::init()
{
	setPurpose("gan generator for eis spectra");
	setInputLabel("RAND");
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

std::shared_ptr<torch::nn::Module> GANGeneratorImpl::operator[](size_t index)
{
	return model->ptr(index);
}

bool GANGeneratorImpl::saveToCheckpointDir(const std::filesystem::path& path)
{
	return true;
}

torch::Tensor GANGeneratorImpl::forward(torch::Tensor x)
{
	return model->forward(x);
}

GANDiscriminatorImpl::GANDiscriminatorImpl(size_t inputSize):
Net(inputSize, 1, false)
{
	init();
}

GANDiscriminatorImpl::GANDiscriminatorImpl(const Json::Value& node):
Net(node)
{

}

void GANDiscriminatorImpl::init()
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

std::pair<torch::Tensor, torch::Tensor> GANDiscriminatorImpl::forwardSplit(torch::Tensor x)
{
	torch::Tensor features = feature->forward(x);
	torch::Tensor out = classifier->forward(features);
	return std::pair<torch::Tensor, torch::Tensor>(out, features);
}

torch::Tensor GANDiscriminatorImpl::forward(torch::Tensor x)
{
	return forwardSplit(x).first;
}

std::shared_ptr<torch::nn::Module> GANDiscriminatorImpl::operator[](size_t index)
{
	if(index < feature->size())
		return feature->ptr(index);
	else
		return classifier->ptr(index);
}

bool GANDiscriminatorImpl::saveToCheckpointDir(const std::filesystem::path& path)
{
	if(!std::filesystem::is_directory(path))
		std::filesystem::create_directories(path);
	if(!std::filesystem::is_directory(path))
		return false;

	Json::Value networkMetadata;
	getConfiguration(networkMetadata);
	std::ofstream networkMetadataFile;
	networkMetadataFile.open(path/"meta.json", std::ios_base::out);
	if(!networkMetadataFile.is_open())
		return false;
	Json::StreamWriterBuilder builder;
	const std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
	writer->write(networkMetadata, &networkMetadataFile);

	networkMetadataFile.close();
	torch::save(shared_from_this(), path/"weights.pt");
	return true;
}

