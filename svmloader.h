#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
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

	svm_problem getSvmProblem(size_t maximum = SIZE_MAX)
	{
		Log(Log::INFO)<<"Construting svm problem:";

		std::random_device rd;
		std::default_random_engine rande(rd());

		svm_problem problem;
		size_t dataSetSize = _dataset->size().value();
		problem.l = maximum < dataSetSize ? maximum : dataSetSize;
		problem.y = new double[problem.l];
		problem.x = new svm_node*[problem.l];
		torch::Tensor exampleZero = _dataset->get(0).data;
		size_t exampleElements = exampleZero.numel();
		for(int i = 0; i < problem.l; ++i)
		{
			torch::data::Example<torch::Tensor, torch::Tensor> example =
				maximum < dataSetSize ? _dataset->get(rande() % dataSetSize) : _dataset->get(i);
			problem.x[i] = new svm_node[exampleElements+1];

			torch::Tensor data = example.data.reshape({static_cast<int64_t>(exampleElements)});
			float* dataPtr = data.contiguous().data_ptr<float>();
			for(size_t j = 0; j < exampleElements; ++j)
			{
				problem.x[i][j].index = j+1;
				problem.x[i][j].value = dataPtr[j];
			}

			problem.x[i][exampleElements].index=-1;//state the end of data vector
			problem.x[i][exampleElements].value=0;

			problem.y[i] = example.target[0].item().to<double>()+1;
			if(i % 100 == 0)
			{
				Log(Log::INFO)<<i<<" of "<<problem.l;
			}
		}
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
