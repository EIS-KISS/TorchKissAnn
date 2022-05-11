#include <iostream>
#include <torch/torch.h>
#include <eisgenerator/model.h>
#include <memory>

#include "model.h"
#include "eisgeneratorloader.h"
#include "testgenerator.h"

void train()
{
	std::cout<<__func__<<'\n';
	std::shared_ptr<Net> net(new Net);

	TestDataset dataset;

	torch::data::DataLoaderOptions options;
	options = options.batch_size(30).workers(16);
	auto dataLoader = torch::data::make_data_loader(dataset.map(torch::data::transforms::Stack<>()), options);

	//Instantiate an SGD optimization algorithm to update our Net's parameters.
	torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);

	for (size_t epoch = 1; epoch <= 100; ++epoch)
	{
		size_t batch_index = 0;
		// Iterate the data loader to yield batches from the dataset.
		for (auto& batch : *dataLoader)
		{
			//std::cout<<"batch "<<typeid(batch).name()<<'\n';
			// Reset gradients.
			optimizer.zero_grad();
			// Execute the model on the input data.
			torch::Tensor prediction = net->forward(batch.data);
			// Compute a loss value to judge the prediction of our model.
 			torch::Tensor loss = torch::l1_loss(prediction, batch.target);
			// Compute gradients of the loss w.r.t. the parameters of our model.
			loss.backward();
			// Update the parameters based on the calculated gradients.
			optimizer.step();
			// Output the loss and checkpoint every 100 batches.
			if (++batch_index % 1000 == 0)
			{
				std::cout << "Epoch: "<<epoch<<" | Batch: "<< batch_index<<" | Loss: "<<loss.item<float>()<<" | Last Prediction "<<prediction[0]<<'\n';
				torch::save(net, "net.pt");
			}
		}
	}
}

int main(int argc, char** argv)
{
	train();

	return 0;
}
