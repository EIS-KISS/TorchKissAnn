/* * TorchKissAnn - A collection of tools to train various types of Machine learning
 * algorithms on various types of EIS data
 * Copyright (C) 2025 Carl Klemm <carl@uvos.xyz>
 *
 * This file is part of TorchKissAnn.
 *
 * TorchKissAnn is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * TorchKissAnn is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with TorchKissAnn.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <cassert>
#include <memory>
#include <vector>
#include "net.h"

namespace ann
{

class AutoEncoder: public Net
{
private:

	std::filesystem::path loadPath;

protected:
	void loadModule(const std::filesystem::path& scriptPath);

	std::shared_ptr<Net> encoder;
	std::shared_ptr<Net> decoder;

public:

	AutoEncoder(const Json::Value& node);
	AutoEncoder(std::shared_ptr<Net> encoder, std::shared_ptr<Net> decoder);

	virtual torch::Tensor forward(torch::Tensor x);

	torch::Tensor forward(torch::Tensor x, torch::Tensor& latent);

	virtual void getConfiguration(Json::Value& node);
	virtual bool saveToCheckpointDir(const std::filesystem::path& path);
	virtual bool loadWeightsFromDir(const std::filesystem::path& path);
	virtual std::shared_ptr<torch::nn::Module> operator[](size_t index){assert(false);};

	virtual void eval();
	virtual void train(bool on = true);
};

}
