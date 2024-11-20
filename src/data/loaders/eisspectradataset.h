#pragma once
#include <kisstype/spectra.h>

class EisSpectraDataset
{
	inline static const std::string extraInputStr = "exip_";

protected:
	static std::pair<std::vector<std::string>, std::vector<std::string>> getExtraInputsAndLabelNames(const eis::Spectra& spectra);
	virtual eis::Spectra loadSpectraAtIndex(size_t index) = 0;
	virtual eis::Spectra loadSpectraHeaderAtIndex(size_t index) = 0;
	virtual std::vector<std::pair<std::string, int64_t>> extraInputs();
	torch::Tensor frequencies();
};
