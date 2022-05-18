#pragma once

class NearestBase
{
protected:
	NearestBase();
	torch::Tensor distM(torch::Tensor a, torch::Tensor b, int power = 2);

};

class NearestNeighbor : public NearestBase
{
	int _power;
	torch::Tensor _data;
	torch::Tensor _labels;

public:
	NearestNeighbor(torch::Tensor data, torch::Tensor lables, int power = 2);
	NearestNeighbor(int power = 2);
	void train(torch::Tensor data, torch::Tensor lables);
	torch::Tensor predict(const torch::Tensor& input);
};
