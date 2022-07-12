#pragma once

template <typename DatasetSelf>
class CloneableDataset :
public torch::data::datasets::Dataset<CloneableDataset<DatasetSelf>, torch::data::Example<torch::Tensor, torch::Tensor>>
{
public:
	virtual CloneableDataset<DatasetSelf>* clone() = 0;
};
