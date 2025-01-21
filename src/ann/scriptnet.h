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

#include <torch/script.h>
#include <cassert>
#include <exception>
#include "net.h"

namespace ann
{

class ScriptNet: public Net
{
public:
	class load_errror: public std::exception
	{
		std::string whatStr;
	public:
		load_errror(const std::string& whatIn): whatStr(whatIn)
		{}
		virtual const char* what() const noexcept override
		{
			return whatStr.c_str();
		}
	};

private:

	std::filesystem::path loadPath;

protected:
	torch::jit::script::Module jitModule;

	void loadModule(const std::filesystem::path& scriptPath);

public:

	ScriptNet(const Json::Value& node, bool noload = true);
	ScriptNet(const std::filesystem::path& scriptPath, bool softmaxI = true, int64_t inputSize = -1, int64_t outputSize = -1);

	virtual torch::Tensor forward(torch::Tensor x);

	virtual void getConfiguration(Json::Value& node);
	virtual bool saveToCheckpointDir(const std::filesystem::path& path);
	virtual bool loadWeightsFromDir(const std::filesystem::path& path);
	virtual std::shared_ptr<torch::nn::Module> operator[](size_t index){assert(false);};

	virtual void eval();
	virtual void train(bool on = true);
};

}
