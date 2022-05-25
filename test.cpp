#include <iostream>
#include <eisgenerator/model.h>
#include <libsvm/svm.h>
#include <random>

#include "eisgendata.h"
#include "nearest.h"
#include "log.h"
#include "svmloader.h"

void testLoader(EisGeneratorDataset* data)
{
	torch::data::DataLoaderOptions options;
	options = options.batch_size(100);
	options = options.workers(16);
	options = options.max_jobs(32);

	std::cout<<options.batch_size()<<' '<<options.workers()<<' '<<options.max_jobs().value()<<' '<<data->is_stateful<<'\n';
	auto dataLoader = torch::data::make_data_loader(data->map(torch::data::transforms::Stack<>()), options);
	size_t i = 0;
	for(auto& batch : *dataLoader)
	{
		std::cout<<"batchnum: "<<++i<<" size "<<batch.data.numel()<<std::endl;
	}
}

int main(int argc, char** argv)
{
	Log::level = Log::DEBUG;

	std::cout<<std::setprecision(3)<<std::fixed<<std::setw(3);

	EisGeneratorDataset* eisData = EisGeneratorDataset::getNeisGeneratedDataset(100);

	return 0;

	testLoader(eisData);
	torch::TensorOptions options;
	options = options.dtype(torch::kFloat32);
	options = options.layout(torch::kStrided);
	options = options.device(torch::kCPU);
	torch::Tensor data = torch::empty({5,3}, options);
	torch::Tensor lable = torch::empty({5}, options);

	for(int64_t i = 0; i < data.size(0); ++i)
	{
		for(int64_t j = 0; j < data.size(1); ++j)
			data[i][j] = static_cast<float>(j == 0? i : j);
	}

	for(int64_t i = 0; i < lable.size(0); ++i)
	{
			lable[i] = static_cast<float>(i);
	}

	NearestNeighbor nearest(data, lable, 2);

	torch::Tensor test = torch::empty({1,3}, options);
	test[0][0] = 1;
	test[0][1] = 1;
	test[0][2] = 2;

	std::cout<<nearest.predict(test);

	return 0;
}
