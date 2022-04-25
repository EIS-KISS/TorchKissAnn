#include <iostream>
#include <torch/torch.h>
#include <eisgenerator/model.h>
#include <memory>

#include "eisgeneratorloader.h"

struct Net : torch::nn::Module
{
	Net()
	{
		// Construct and register two Linear submodules.
		fc1 = register_module("fc1", torch::nn::Linear(50, 30));
		fc2 = register_module("fc2", torch::nn::Linear(30, 15));
		fc3 = register_module("fc3", torch::nn::Linear(15, 5));
	}

	torch::Tensor forward(torch::Tensor x)
	{
		// Use one of many tensor manipulation functions.
		x = torch::relu(fc1->forward(x));
		x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
		x = torch::relu(fc2->forward(x));
		x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
		return x;
	}

	// Use one of many "standard library" modules.
	torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

void train()
{
	std::cout<<__func__<<'\n';
	std::shared_ptr<Net> net(new Net);

	EisGeneratorDataset dataset;

	torch::data::DataLoaderOptions options;
	options = options.batch_size(64).workers(0);
	auto dataLoader = torch::data::make_data_loader(dataset.map(torch::data::transforms::Stack<>()), options);
	std::cout<<"dataloader "<<typeid(dataLoader).name()<<'\n';

	//Instantiate an SGD optimization algorithm to update our Net's parameters.
	torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);

	for (size_t epoch = 1; epoch <= 10; ++epoch)
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
 			torch::Tensor loss = torch::nll_loss(prediction, batch.target);
			// Compute gradients of the loss w.r.t. the parameters of our model.
			loss.backward();
			// Update the parameters based on the calculated gradients.
			optimizer.step();
			// Output the loss and checkpoint every 100 batches.
			if (++batch_index % 100 == 0)
			{
				std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
							<< " | Loss: " << loss.item<float>() << std::endl;
				// Serialize your model periodically as a checkpoint.
				torch::save(net, "net.pt");
			}
		}
	}
}

int main(int argc, char** argv)
{
	EisGeneratorDataset dataset;
	torch::data::Example<torch::Tensor, torch::Tensor> example = dataset.get(0);

	std::cout<<"data: "<<example.data<<'\n'<<"target: "<<example.target<<'\n';
	std::cout<<"data: "<<example.data.dim()<<'\n'<<"target: "<<example.target.dim()<<'\n';

	train();

	return 0;
}
