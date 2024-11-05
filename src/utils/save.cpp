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
