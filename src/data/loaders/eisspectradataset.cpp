#include "eisspectradataset.h"
#include "data/eistotorch.h"

std::pair<std::vector<std::string>, std::vector<std::string>> EisSpectraDataset::getExtraInputsAndLabelNames(const eis::Spectra& spectra)
{
	std::vector<std::string> labelNames;
	std::vector<std::string> extraInputNames;
	for(const std::string& label : spectra.labelNames)
	{
		if(label.find(extraInputStr) == 0)
			extraInputNames.push_back(label);
		else
			labelNames.push_back(label);
	}

	return {extraInputNames, labelNames};
}

std::vector<std::pair<std::string, int64_t>> EisSpectraDataset::extraInputs()
{
	eis::Spectra spectra = loadSpectraAtIndex(0);
	std::vector<std::pair<std::string, int64_t>> out;
	for(const std::string& name : getExtraInputsAndLabelNames(spectra).first)
	{
		out.push_back({name, 1});
		out.back().first.erase(out.back().first.begin(), out.back().first.begin()+extraInputStr.length());
	}
	return out;
}

torch::Tensor EisSpectraDataset::frequencies()
{
	eis::Spectra spectra = loadSpectraAtIndex(0);
	std::vector<fvalue> omega(spectra.data.size());
	for(size_t i = 0; i < spectra.data.size(); ++i)
		omega[i] = spectra.data[i].omega;
	return fvalueVectorToTensor(omega);
}
