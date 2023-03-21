#include "tensoroperators.h"

#include <fstream>
#include <sstream>

#include "log.h"

torch::Tensor toeplitz(const torch::Tensor& b, const torch::Tensor a)
{
	torch::Tensor indeciesB = torch::linspace(1, b.numel()-1, b.numel()-1, torch::TensorOptions().dtype(torch::kInt32));
	torch::Tensor line = torch::cat({a.reshape({1, a.numel()}), torch::index_select(b.reshape({1, b.numel()}), 1, indeciesB).flip(1)}, 1);
	torch::Tensor output = line;
	for(int64_t i = 1; i < b.numel(); ++i)
		output = torch::cat({output, torch::roll(line, i)}, 0);

	torch::Tensor indeciesCols = torch::linspace(0, a.numel()-1, a.numel(), torch::TensorOptions().dtype(torch::kInt32));
	return torch::index_select(output, 1, indeciesCols);
}

torch::Tensor fnToTensor(std::function<double(double)> fn, double start, double stop, size_t size)
{
	torch::Tensor out = torch::empty({static_cast<int64_t>(size)}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
	auto outA = out.accessor<float, 1>();

	double step = (stop-start)/(size-1);

	for(size_t i = 0; i < size; ++i)
		outA[i] = fn(start+i*step);
	return out;
}

double tensorSimpleIntegrate(torch::Tensor in, double length)
{
	return (torch::mean(in).item().to<double>())*length;
}

double integrateFn(std::function<double(double)> fn, double start, double stop, size_t steps, double *prev)
{
	double resH =  prev ? *prev : tensorSimpleIntegrate(fnToTensor(fn, start, stop, steps/2), stop-start);
	double res = tensorSimpleIntegrate(fnToTensor(fn, start, stop, steps), stop-start);

	double relDiff = res/resH;
	if((relDiff > 0.99 && relDiff < 1.01) || steps > 3e6)
		return res;
	else
		return integrateFn(fn, start, stop, steps*2, &res);
}

torch::Tensor unargmax(const torch::Tensor& in, int64_t size)
{
	assert(size >= 0);
	assert(in.dim() == 1);
	torch::Tensor multiClassTargets = torch::zeros({in.size(0), size}, in.options());

	for(int i = 0; i < in.size(0); ++i)
	{
		multiClassTargets[i][in[i].template item<int64_t>()] = 1;
	}
	return multiClassTargets;
}

template<typename DataType>
static void save2dTensorToCsv(const torch::Tensor& tensor, std::iostream& stream)
{
	auto tensorAcesssor = tensor.accessor<DataType, 2>();
	for(int i = 0; i < tensorAcesssor.size(0); ++i)
	{
		for(int j = 0; j < tensorAcesssor.size(1); ++j)
		{
			stream<<tensorAcesssor[i][j];
			if(j+1 < tensorAcesssor.size(1))
				stream<<", ";
		}
		stream<<'\n';
	}
}

template<typename DataType>
static void save1dTensorToCsv(const torch::Tensor& tensor, std::iostream& stream)
{
	auto tensorAcesssor = tensor.accessor<DataType, 1>();

	for(int i = 0; i < tensorAcesssor.size(0); ++i)
	{
		stream<<tensorAcesssor[i];
		if(i+1 < tensorAcesssor.size(0))
			stream<<", ";
	}
	stream<<'\n';
}

bool saveTensorToCsv(const std::filesystem::path& path, const torch::Tensor& tensor, const std::string& name)
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

std::string tensorToString(const torch::Tensor& tensor)
{
	std::stringstream ss;
	ss<<std::scientific;

	if(tensor.dim() == 2)
	{
		if(tensor.dtype() == torch::kFloat32)
			save2dTensorToCsv<float>(tensor, ss);
		else
			save2dTensorToCsv<int64_t>(tensor, ss);
	}
	else if(tensor.dim() == 1)
	{
		if(tensor.dtype() == torch::kFloat32)
			save1dTensorToCsv<float>(tensor, ss);
		else
			save1dTensorToCsv<int64_t>(tensor, ss);
	}
	else
	{
		Log(Log::ERROR)<<"cant convert "<<tensor.dim()<<" dimentional tensor to string\n";
		return "";
	}
	return ss.str();
}

