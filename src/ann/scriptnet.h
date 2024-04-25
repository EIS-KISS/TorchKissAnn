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
