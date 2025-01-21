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

#include "autoencoder.h"
#include "log.h"

#include <json/reader.h>
#include <json/json.h>
#include <filesystem>
#include <stdexcept>
#include <fstream>


ann::AutoEncoder::AutoEncoder(const Json::Value& node):
Net(node)
{
}

ann::AutoEncoder::AutoEncoder(std::shared_ptr<Net> encoder, std::shared_ptr<Net> decoder):
Net(encoder->getInputSize(), decoder->getOutputSize(), decoder->hasSoftmaxOutput()),
encoder(encoder),
decoder(decoder)
{
	if(encoder->getOutputSize() != decoder->getInputSize())
		throw std::invalid_argument("encoder output must be the same shape as decoder output");
	torch::nn::Module::register_module("encoder", encoder);
	torch::nn::Module::register_module("decoder", decoder);
}

torch::Tensor ann::AutoEncoder::forward(torch::Tensor x, torch::Tensor& latent)
{
	latent = encoder->forward(x);
	torch::Tensor out = decoder->forward(latent);
	return out;
}

torch::Tensor ann::AutoEncoder::forward(torch::Tensor x)
{
	torch::Tensor latent;
	return forward(x, latent);
}

bool ann::AutoEncoder::saveToCheckpointDir(const std::filesystem::path &path)
{
	if(!std::filesystem::is_directory(path))
		std::filesystem::create_directories(path);
	if(!std::filesystem::is_directory(path/"encoder"))
		std::filesystem::create_directories(path/"encoder");
	if(!std::filesystem::is_directory(path/"decoder"))
		std::filesystem::create_directories(path/"decoder");
	if(!std::filesystem::is_directory(path/"encoder"))
	{
		Log(Log::ERROR)<<"Unable to create "<<path/"encoder";
		return false;
	}
	if(!std::filesystem::is_directory(path/"decoder"))
	{
		Log(Log::ERROR)<<"Unable to create "<<path/"decoder";
		return false;
	}

	Json::Value networkMetadata;
	getConfiguration(networkMetadata);

	Json::Value encoderMeta;
	encoder->getConfiguration(encoderMeta);
	Json::Value decoderMeta;
	decoder->getConfiguration(decoderMeta);

	networkMetadata["encoder"] = encoderMeta;
	networkMetadata["decoder"] = decoderMeta;

	std::ofstream networkMetadataFile;
	networkMetadataFile.open(path/"meta.json", std::ios_base::out);
	if(!networkMetadataFile.is_open())
		return false;
	Json::StreamWriterBuilder builder;
	const std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
	writer->write(networkMetadata, &networkMetadataFile);
	networkMetadataFile.close();

	Json::FastWriter fastWriter;
	torch::jit::ExtraFilesMap files;
	files["meta.json"] = fastWriter.write(networkMetadata);

	encoder->saveToCheckpointDir(path/"encoder");
	decoder->saveToCheckpointDir(path/"decoder");
	return true;
}

bool ann::AutoEncoder::loadWeightsFromDir(const std::filesystem::path& path)
{
	encoder = ann::Net::newNetFromCheckpointDir(path/"encoder");
	decoder = ann::Net::newNetFromCheckpointDir(path/"decoder");
	torch::nn::Module::register_module("encoder", encoder);
	torch::nn::Module::register_module("decoder", decoder);
	return encoder && decoder;
}

void ann::AutoEncoder::getConfiguration(Json::Value& node)
{
	Net::getConfiguration(node);
	node["type"] = typeid(*this).name();
}

void ann::AutoEncoder::eval()
{
	train(true);
}

void ann::AutoEncoder::train(bool on)
{
	encoder->train(on);
	decoder->train(on);
}
