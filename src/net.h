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

#include <cstdint>
#include <vector>
#include <json/json.h>
#include <filesystem>

#include "log.h"

namespace ann
{

class Net : public torch::nn::Module
{
public:
	 // Smallest magnitude the signed log input compression still resolves. Impedance magnitudes
	 // below this are mapped to zero, which keeps the noise floor of a measurement from being
	 // amplified into the same range as the signal.
	static constexpr double DEFAULT_INPUT_LOG_FLOOR = 1e-9;

protected:
	int64_t inputSize;
	int64_t outputSize;
	std::string inputLabel = "EIS";
	torch::Tensor inputFrequencies;
	std::vector<std::string> outputLabels;
	std::string purpose = "Unkown";
	std::vector<std::pair<std::string, int64_t>> extraInputs;
	torch::Tensor outputScalars;
	torch::Tensor outputBiases;
	torch::Tensor outputLogMask; //Entries with a set flag are interpreted as the exponent of a base ten power instead of as the value itself.
	torch::Tensor inputScalars; // Affine transform applied to the raw dataset input before it reaches the network. Empty means the input is passed through unchanged.
	torch::Tensor inputBiases;
	int64_t inputLogCount = 0;
	double inputLogFloor = DEFAULT_INPUT_LOG_FLOOR;
	bool softmax;

public:
	Net(const Json::Value& node);
	Net(int64_t inputSizeI, int64_t outputSizeI, bool softmaxI = true);

	virtual torch::Tensor forward(torch::Tensor x) = 0;

	virtual int64_t getInputSize() const;
	virtual int64_t getOutputSize() const;
	bool hasSoftmaxOutput() const;
	const std::string& getPurpose() const;
	void setPurpose(const std::string& str);
	const std::vector<std::string>& getOutputLabels() const;
	void setOutputLabels(const std::vector<std::string>& outputLabels);
	void setInputLabel(const std::string& inputLabel);
	void setOutputScalars(const torch::Tensor& tensor);
	void setOutputBiases(const torch::Tensor& tensor);
	void setOutputLogMask(const torch::Tensor& tensor);
	void setInputFrequencies(const torch::Tensor& tensor);
	/**
	 * Configure how a raw dataset input is turned into what the network sees. Empty scalars disable
	 * the transform. logCount features at the front of the input vector are compressed with a
	 * signed log first, which is what the spectral parts of an EIS input need.
	 */
	void setInputNormalization(const torch::Tensor& scalars, const torch::Tensor& biases,
			int64_t logCount = 0, double logFloor = DEFAULT_INPUT_LOG_FLOOR);
	torch::Tensor getOutputScalars();
	torch::Tensor getOutputBiases();
	torch::Tensor getOutputLogMask();
	torch::Tensor getInputScalars();
	int64_t getInputLogCount() const;
	double getInputLogFloor() const;
	/**
	 * Transform a raw dataset input into the space the network was trained in.
	 */
	torch::Tensor normalizeInput(torch::Tensor input) const;
	/**
	 * Transform physical targets into the space the network was trained in.
	 */
	torch::Tensor normalizeTarget(torch::Tensor target) const;
	/**
	 * Transform a raw network output into physical values.
	 */
	torch::Tensor denormalizeOutput(torch::Tensor output) const;
	void setExtraInputs(const std::vector<std::pair<std::string, int64_t>>& in);
	std::vector<std::pair<std::string, int64_t>> getExtraInputs();
	virtual void getConfiguration(Json::Value& node);
	virtual bool saveToCheckpointDir(const std::filesystem::path& path);
	virtual bool loadWeightsFromDir(const std::filesystem::path& path);
	virtual std::shared_ptr<torch::nn::Module> operator[](size_t index) = 0;
	static std::shared_ptr<Net> newNetFromConfiguation(const Json::Value& node);
	static std::shared_ptr<Net> newNetFromCheckpointDir(const std::filesystem::path& path);
	template <typename DatasetType> bool setOutputLabelsFromDataset(DatasetType* dataset);
};

template <typename DatasetType> bool Net::setOutputLabelsFromDataset(DatasetType* dataset)
{
	if(getOutputSize() != static_cast<int64_t>(dataset->outputSize()))
		Log(Log::WARN)<<"This dataset has the wrong number of classes, cant set the output lables from this dataset";
	std::vector<std::string> outputLablesTmp(getOutputSize());
	for(int64_t i = 0; i < getOutputSize(); ++i)
	{
		std::string label = dataset->modelStringForClass(i);
		if(label == "Unkown")
		{
			Log(Log::WARN)<<"This dataset dosent provide output labels";
			return false;
		}
		outputLablesTmp[i] = dataset->modelStringForClass(i);
	}
	setOutputLabels(outputLablesTmp);
	return true;
}

}
