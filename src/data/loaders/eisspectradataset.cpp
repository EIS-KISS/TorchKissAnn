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
