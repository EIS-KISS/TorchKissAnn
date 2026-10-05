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

	static constexpr double INPUT_FEATURE_CLAMP = 2.0;

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
	std::string spectraNorm; // Name of the per spectra |Z| scaling the loader applied to every training input, empty when the inputs are raw impedances. A client feeding the network has to reproduce it.
	bool canonicalGrid = false; // The network expects inputs resampled onto a fixed log frequency grid.
	double canonicalFMin = 0;
	double canonicalFMax = 0;
	int64_t canonicalPoints = 0;
	bool canonicalLog = true;
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
	// Record that the network was trained on inputs resampled onto a fixed canonical frequency grid.
	// libkissinference needs fMin, fMax, points and log to reproduce the resampling (linear
	// interpolation with constant extrapolation, matching eis::fitToFrequencies) before feeding the
	// fixed size graph.
	void setCanonicalGrid(double fMin, double fMax, int64_t points, bool log);
	bool hasCanonicalGrid() const;
	double getCanonicalFMin() const;
	double getCanonicalFMax() const;
	int64_t getCanonicalPoints() const;
	bool getCanonicalLog() const;
	// Configure how a raw dataset input is turned into what the network sees. Empty scalars disable
	// the transform. logCount features at the front of the input vector are compressed with a
	// signed log first, which is what the spectral parts of an EIS input need.
	void setInputNormalization(const torch::Tensor& scalars, const torch::Tensor& biases,
			int64_t logCount = 0, double logFloor = DEFAULT_INPUT_LOG_FLOOR);
	// Compress the leading logCount input features with a signed log but apply no affine rescaling.
	// Classification uses this: the per spectra |Z| normalization already puts every input on a common
	//scale, so an affine transform derived from the dataset range would carry no information.
	void setInputLogCompression(int64_t logCount, double logFloor = DEFAULT_INPUT_LOG_FLOOR);
	// Record that the training loader scaled every spectra to a common impedance level. mode is one of
	// the names understood by SpectraNorm (zmean, zgeom, zmax); an empty string means the network was
	// trained on raw impedances. Anything feeding this network has to apply the same scaling, after
	// the canonical grid resampling and before normalizeInput.
	void setSpectraNorm(const std::string& mode);

	// Name of the per spectra |Z| scaling this network expects, empty when there is none.
	const std::string& getSpectraNorm() const;
	torch::Tensor getOutputScalars();
	torch::Tensor getOutputBiases();
	torch::Tensor getOutputLogMask();

	torch::Tensor normalizeInput(torch::Tensor input) const;
	torch::Tensor normalizeTarget(torch::Tensor target) const;
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
