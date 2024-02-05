#include "gan.h"

#include <cstdint>

#include "log.h"
#include "simplenet.h"

using namespace gan;

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> anomalyScore(torch::Tensor data, torch::Tensor fake, GANDiscriminator &dis, float lambda = 0.1)
{
    torch::Tensor resLoss = torch::abs(data - fake).sum();
    torch::Tensor feature = dis->forwardSplit(data).second;
    torch::Tensor fakeFeature = dis->forwardSplit(fake).second;
    torch::Tensor disLoss = torch::abs(feature - fakeFeature).sum();
    torch::Tensor score = (1.0 - lambda) * resLoss + lambda * disLoss;
    return {score, resLoss, disLoss};
}

torch::Tensor gan::generate(const std::string& fileName)
{
	GANGenerator gen(Z_SIZE, DATA_WIDTH);
	torch::load(gen, fileName + ".gen.vos");
	return gen->forward(torch::randn({1, Z_SIZE}));
}

bool gan::filter(torch::Tensor input, const std::string& fileName, float nabla)
{
	GANGenerator gen(Z_SIZE, DATA_WIDTH);
	GANDiscriminator dis(DATA_WIDTH);
	torch::load(dis, fileName + std::string(".dis.vos"));
	torch::load(gen, fileName + std::string(".gen.vos"));
	dis->eval();
	gen->eval();

	torch::Tensor fake = gen->forward(torch::randn({1, Z_SIZE}));

	auto [score, resLoss, disLoss] = anomalyScore(input, fake, dis);
	Log(Log::DEBUG)<<"score: "<<score.item<float>();
	return score.item<float>() < nabla;
}
