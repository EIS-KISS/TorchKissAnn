#include "nearest.h"
#include <cstdint>
#include "log.h"

NearestBase::NearestBase()
{}

torch::Tensor NearestBase::distM(torch::Tensor a, torch::Tensor b, int power)
{
	int64_t aSizeA = a.size(0);
	int64_t bSizeA = b.size(0);
	int64_t aSizeB = a.dim() > 1 ? a.size(1) : -1;

	if(aSizeB == -1)
	{
		a = a.unsqueeze(1).expand({aSizeA, bSizeA});
	}
	else
	{
		a = a.unsqueeze(1).expand({aSizeA, bSizeA, aSizeB});
		b = b.unsqueeze(0).expand({aSizeA, bSizeA, aSizeB});
	}

	return at::pow(a-b, power).sum(2);
}

NearestNeighbor::NearestNeighbor(const torch::Tensor data, const torch::Tensor labels, int power):
_power(power)
{
	train(data, labels);
}

NearestNeighbor::NearestNeighbor(int power):
_power(power)
{

}

void NearestNeighbor::train(const torch::Tensor data, const torch::Tensor labels)
{
	_data = data;
	_labels = labels;
}

torch::Tensor NearestNeighbor::predict(const torch::Tensor& input)
{
	if(_data.numel() == 0 || _labels.numel() == 0)
	{
		Log(Log::ERROR)<<"NearestNeighbor model not trained\n";
		return torch::Tensor();
	}

	torch::Tensor distance = distM(input, _data,  _power);
	auto label = distance.argmin(1);
	std::cout<<label<<std::endl;
	return _labels[label[0]];
}
