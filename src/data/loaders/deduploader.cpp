#include "deduploader.h"

#include <algorithm>
#include <eisgenerator/eistype.h>
#include <stdexcept>

#include "csv.h"
#include "mpt.h"
#include "../print.h"
#include "log.h"
#include "tokenize.h"
#include "../filterdata.h"
#include "../eistotorch.h"

using namespace eis;

bool isWhitespace(const char in)
{
	switch(in)
	{
		case ' ':
		case '\t':
		return true;
		default:
		return false;
	}
}

FastDedupDataset::FastDedupDataset(const std::filesystem::path& fileName)
{
	if(!(std::filesystem::is_regular_file(fileName) || std::filesystem::is_symlink(fileName)))
	{
		Log(Log::ERROR)<<fileName<<" dose not exist";
		return;
	}

	file.open(fileName, std::ios_base::in);
	if(!file.is_open())
	{
		Log(Log::ERROR)<<"Could not open "<<fileName;
		return;
	}

	std::string line;
	std::getline(file, line, '\n');
	std::vector<std::string> tokens = tokenize(line, ',', '{', '}');
	for(std::string& token : tokens)
	{
		std::remove_if(token.begin(), token.end(), &isWhitespace);
		if(token.find("class") == 0)
		{
			token.erase(0, 5);
			size_t suggestedMaxclass = std::stoul(token)+1;
			modelCount = modelCount < suggestedMaxclass ? suggestedMaxclass : modelCount;
		}
		else if(token.find("re") == 0)
		{
			token.erase(0, 2);
			size_t suggestedDepth = std::stoul(token)+1;
			depth = depth < suggestedDepth ? suggestedDepth : depth;
		}
	}

	std::istreambuf_iterator<char> iter = std::find(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n');
	while(iter != std::istreambuf_iterator<char>() && file.good())
	{
		iter = std::find(iter, std::istreambuf_iterator<char>(), '\n');
		lineStarts.push_back(file.tellg());
	}

	file.seekg(0, std::ios_base::beg);
	file.clear();

	Log(Log::DEBUG)<<"Loaded dataset from "<<fileName<<" with "<<size().value()<<" examples of "<<modelCount<<" classes and a depth of "<<depth;
}

torch::data::Example<torch::Tensor, torch::Tensor> FastDedupDataset::parseLine(const std::string& line)
{
	torch::TensorOptions options;
	options = options.dtype(torch::kFloat32);
	options = options.layout(torch::kStrided);
	options = options.device(torch::kCPU);
	torch::Tensor input = torch::empty({static_cast<long int>(depth*2)}, options);

	options = options.dtype(torch::kInt8);
	torch::Tensor target = torch::empty({static_cast<long int>(modelCount)}, options);

	std::vector<std::string> tokens = tokenize(line, ',', '{', '}');
	if(tokens.size() != modelCount+depth*2+3)
	{
		const std::string error = std::string("line of file dose expected (") +
			std::to_string(modelCount+depth*2+3) + std::string(") number of fields");
		Log(Log::ERROR)<<error;
		throw std::length_error(error);
	}


	for(size_t i = 3; i < 3+modelCount; ++i)
	{
		size_t j = i-3;
		target[j] = static_cast<uint8_t>(std::stoi(tokens[i]));
	}
	for(size_t i = 3+modelCount; i < 3+modelCount+depth; ++i)
	{
		size_t j = i-(3+modelCount);
		input[j] = std::stof(tokens[i]);
		input[j+depth] = std::stof(tokens[i+depth]);
	}

	return torch::data::Example<torch::Tensor, torch::Tensor>(input, target);
}

torch::data::Example<torch::Tensor, torch::Tensor> FastDedupDataset::get(size_t index)
{
	file.seekg(lineStarts[index]);
	std::string line;
	std::getline(file, line, '\n');
	return parseLine(line);
}

size_t FastDedupDataset::classesCount()
{
	return modelCount;
}

c10::optional<size_t> FastDedupDataset::size() const
{
	return lineStarts.size();
}

torch::Tensor FastDedupDataset::modelWeights()
{
	size_t classes = classesCount();

	torch::TensorOptions options;
	options = options.dtype(torch::kFloat32);
	options = options.layout(torch::kStrided);
	options = options.device(torch::kCPU);
	torch::Tensor output = torch::ones({static_cast<int64_t>(classes)}, options);

	return output;
}
