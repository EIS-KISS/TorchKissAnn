#include "deduploader.h"

#include <algorithm>
#include <eisgenerator/eistype.h>
#include <stdexcept>

#include "mpt.h"
#include "../print.h"
#include "log.h"
#include "tokenize.h"
#include "../filterdata.h"
#include "../eistotorch.h"

using namespace eis;

static bool isWhitespace(const char in)
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

FastDedupDataset::FastDedupDataset(const std::filesystem::path& fileName): path(fileName)
{
	Log(Log::DEBUG)<<"Trying to load "<<fileName;
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

	Log(Log::DEBUG)<<"Reading header";
	std::string line;
	std::getline(file, line, '\n');
	std::vector<std::string> tokens = tokenize(line, ',', '{', '}');
	tokens[0].resize(std::remove_if(tokens[0].begin(), tokens[0].end(), &isWhitespace) - tokens[0].begin());
	if(tokens[0] != "UVOSEISDATATAB")
	{
		Log(Log::ERROR)<<fileName<<" is not a valid uvos eis table file";
		file.close();
		return;
	}
	if(tokens.size() >= 2)
	{
		std::vector<std::string> supertoken = tokenize(tokens[1], '.');
		if(supertoken.size() < 2)
		{
			Log(Log::ERROR)<<fileName<<" can not parse version";
			file.close();
			return;
		}

		versionMajor = std::stoi(supertoken[0]);
		versionMinor = std::stoi(supertoken[1]);
	}
	Log(Log::DEBUG)<<"Got table of version "<<versionMajor<<'.'<<versionMinor;

	std::getline(file, line, '\n');
	tokens = tokenize(line, ',', '{', '}');
	tokens[0].resize(std::remove_if(tokens[0].begin(), tokens[0].end(), &isWhitespace) - tokens[0].begin());
	modelStrings.reserve(tokens.size()-1);
	if(tokens.size() < 2 || tokens[0] != "Models")
	{
		Log(Log::ERROR)<<fileName<<" can not parse models";
		file.close();
		return;
	}
	for(size_t i = 1; i < tokens.size(); ++i)
	{
		tokens[i].resize(std::remove_if(tokens[i].begin(), tokens[i].end(), &isWhitespace) - tokens[i].begin());
		modelStrings.push_back(tokens[i]);
	}

	std::getline(file, line, '\n');
	tokens = tokenize(line, ',', '{', '}');
	for(std::string& token : tokens)
	{
		token.resize(std::remove_if(token.begin(), token.end(), &isWhitespace) - token.begin());
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

	if(modelCount != modelStrings.size())
	{
		Log(Log::ERROR)<<"Number of model strings and model ids dose not match!";
		file.close();
		return;
	}

	Log(Log::DEBUG)<<"Reading lines";
	lineStarts.push_back(file.tellg());
	std::istreambuf_iterator<char> iter = std::find(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n');
	iter++;
	while(iter != std::istreambuf_iterator<char>() && file.good())
	{
		lineStarts.push_back(file.tellg()+std::streamoff(1));
		iter = std::find(iter, std::istreambuf_iterator<char>(), '\n');
		iter++;
	}
	file.seekg(0, std::ios_base::beg);
	file.clear();

	Log(Log::DEBUG)<<"Loaded dataset from "<<fileName<<" with "<<size().value()<<" examples of "<<modelCount<<" classes and a depth of "<<depth;
}

FastDedupDataset::FastDedupDataset(const FastDedupDataset& in)
{
	modelCount = in.modelCount;
	depth = in.depth;
	lineStarts = in.lineStarts;
	modelStrings = in.modelStrings;
	versionMajor = in.versionMajor;
	versionMinor = in.versionMinor;
	classCounts = in.classCounts;
	path = in.path;
	file.open(path, std::ios_base::in);
	if(!file.is_open())
		Log(Log::ERROR)<<"Could not open "<<path;
}

std::string FastDedupDataset::modelStringForClass(size_t classNum)
{
	return modelStrings[classNum];
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
		const std::string error = std::string("line of file dose not contain expected (") +
			std::to_string(modelCount+depth*2+3) + std::string(") number of fields");
		Log(Log::ERROR)<<error;
		Log(Log::ERROR)<<"Line is: "<<line;
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

size_t FastDedupDataset::parseOriginalClass(const std::string& line)
{
	std::vector<std::string> tokens = tokenize(line, ',', '{', '}');

	if(tokens.size() < 3)
	{
		const std::string error = std::string("a line of the file dose not contain at least 3 fields");
		Log(Log::ERROR)<<error;
		Log(Log::ERROR)<<"Line is: "<<line;
		throw std::length_error(error);
	}

	size_t classNum = std::stoul(tokens[2]);
	if(classNum > classesCount())
	{
		const std::string error = std::string("The specified class in ");
		Log(Log::ERROR)<<error;
		Log(Log::ERROR)<<"Line is: "<<line;
		throw std::length_error(error);
	}

	return classNum;
}

torch::data::Example<torch::Tensor, torch::Tensor> FastDedupDataset::getImpl(size_t index)
{
	file.seekg(lineStarts[index]);
	std::string line;
	std::getline(file, line, '\n');
	return parseLine(line);
}


size_t FastDedupDataset::classForIndex(size_t index)
{
	file.seekg(lineStarts[index]);
	std::string line;
	std::getline(file, line, '\n');
	return parseOriginalClass(line);
}

size_t FastDedupDataset::classesCount() const
{
	return modelCount;
}

c10::optional<size_t> FastDedupDataset::size() const
{
	return lineStarts.size();
}

std::vector<size_t> FastDedupDataset::calculateClassCounts()
{
	file.clear();
	std::vector<size_t> classCounts(modelCount);
	for(size_t i = 0; i < lineStarts.size(); ++i)
	{
		file.seekg(lineStarts[i]);
		std::string line;
		std::getline(file, line, '\n');
		std::vector<std::string> tokens = tokenize(line, ',', '{', '}');
		if(tokens.size() < 3)
			throw std::length_error(std::string("line ") + std::to_string(i) + " is to short: " + line);
		size_t classNum = std::stoul(tokens[2]);
		if(classNum > classCounts.size()-1)
			throw std::length_error(std::string("line ") + std::to_string(i) + " has originClass that is out of range");
		++classCounts[classNum];
	}
	return classCounts;
}

torch::Tensor FastDedupDataset::classWeights()
{
	size_t classes = classesCount();

	torch::TensorOptions options;
	options = options.dtype(torch::kFloat32);
	options = options.layout(torch::kStrided);
	options = options.device(torch::kCPU);
	torch::Tensor output = torch::empty({static_cast<int64_t>(classes)}, options);

	if(classCounts.size() != modelCount)
		classCounts = calculateClassCounts();

	for(size_t i = 0; i < modelCount; ++i)
		output[i] = size().value()/static_cast<float>(classCounts[i]);

	//output = torch::nn::functional::normalize(output, torch::nn::functional::NormalizeFuncOptions().p(2).dim(0));

	return output;
}

