#include <iostream>
#include <torch/torch.h>
#include <eisgenerator/model.h>

#include "eisgeneratorloader.h"

int main(int argc, char** argv)
{
	EisGeneratorDataset dataset;
	size_t size = dataset.size().value();
	std::cout<<"dataset size: "<<size<<'\n';
	for(size_t i = 0; i < 10; ++i)
	{
		torch::data::Example<torch::Tensor, torch::Tensor> example = dataset.get(i*((size-1)/10));
		std::cout<<"data index "<<i*((size-1)/10)<<":\n";
		std::cout<<example.data;
		std::cout<<"\ntarget:\n";
		std::cout<<example.target<<'\n';
	}

	torch::data::Example<torch::Tensor, torch::Tensor> example = dataset.get(size/2);
	std::cout<<"outptut data dim: "<<example.data.dim()<<" output data size[0]: "<<example.data.size(0)<<'\n';
	std::cout<<"outptut target dim: "<<example.target.dim()<<" output target size[0]: "<<example.target.size(0)<<'\n';

	return 0;
}
