#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <iostream>
#include <libsvm/svm.h>
#include "log.h"

template <typename DatasetSelf>
class SvmLoader
{
	torch::data::datasets::Dataset<DatasetSelf, torch::data::Example<torch::Tensor, torch::Tensor>>* _dataset;
public:

	SvmLoader(torch::data::datasets::Dataset<DatasetSelf, torch::data::Example<torch::Tensor, torch::Tensor>>* dataset): _dataset(dataset)
	{

	}

	svm_problem getSvmProblem(size_t maximum = SIZE_MAX, bool oneClass = false)
	{
		std::random_device rd;
		std::default_random_engine rande(rd());

		svm_problem problem;
		size_t dataSetSize = _dataset->size().value();
		problem.l = maximum < dataSetSize ? maximum : dataSetSize;
		problem.y = new double[problem.l];
		problem.x = new svm_node*[problem.l];

		Log(Log::INFO)<<"Construting svm problem of size "<<problem.l<<":";

		for(int i = 0; i < problem.l; ++i)
		{
			torch::data::Example<torch::Tensor, torch::Tensor> example =
				maximum < dataSetSize ? _dataset->get(rande() % dataSetSize) : _dataset->get(i);
			problem.x[i] = tensorToNewSvmNodeArray(example.data);

			problem.y[i] = oneClass ? 0 : example.target[0].item().to<double>();
			if(i % 100 == 0)
			{
				if(Log::level == Log::DEBUG)
				{
					Log(Log::INFO)<<i<<" of "<<problem.l;
					std::cout<<std::flush;
				}
				else
				{
					Log(Log::INFO, false)<<'.';
					std::cout<<std::flush;
				}
			}
		}
		if(Log::level != Log::DEBUG)
			std::cout<<std::endl;
		return problem;
	}

	static void distroySvmProblem(svm_problem problem)
	{
		delete[] problem.y;
		for(int i = 0; i < problem.l; ++i)
		{
			delete[] problem.x[i];
		}
		delete[] problem.x;
	}

	static svm_node* tensorToNewSvmNodeArray(torch::Tensor tensor)
	{
		tensor = tensor.reshape({1, tensor.numel()});
		float* dataPtr = tensor.contiguous().data_ptr<float>();
		svm_node *nodeArray = new svm_node[tensor.numel()+1];
		for(int64_t i = 0; i < tensor.numel(); ++i)
		{
			nodeArray[i].index = i;
			nodeArray[i].value = dataPtr[i];
		}
		nodeArray[tensor.numel()].index = -1;
		nodeArray[tensor.numel()].value = 0;
		return nodeArray;
	}
};

void printSvmNode(const svm_node* nodeArray);
void printSvmProblem(svm_problem problem);
void svmPrintFunction(const char* str);
void setSvmWeights(svm_parameter* params, const torch::Tensor& weigts);
