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

#include "save.h"

bool csv::save(const std::filesystem::path& path, const torch::Tensor& tensor, const std::string& name)
{
	std::fstream file;
	file.open(path, std::ios_base::out);
	if(!file.is_open())
	{
		Log(Log::ERROR)<<"can not open "<<path<<" for writing\n";
		return false;
	}

	file<<std::scientific;
	file<<name<<'\n';
	if(tensor.dim() == 2)
	{
		if(tensor.dtype() == torch::kFloat32)
			save2dTensorToCsv<float>(tensor, file);
		else
			save2dTensorToCsv<int64_t>(tensor, file);
	}
	else if(tensor.dim() == 1)
	{
		if(tensor.dtype() == torch::kFloat32)
			save1dTensorToCsv<float>(tensor, file);
		else
			save1dTensorToCsv<int64_t>(tensor, file);
	}
	else
	{
		Log(Log::ERROR)<<"can not save "<<tensor.dim()<<" dimentional tensor\n";
		return false;
	}
	file.close();
	return true;
}
