//
// TorchKissAnn - A collection of tools to train various types of Machine learning
// algorithms on various types of EIS data
// Copyright (C) 2025 Carl Klemm <carl@uvos.xyz>
//
// This file is part of TorchKissAnn.
//
// TorchKissAnn is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// TorchKissAnn is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with TorchKissAnn.  If not, see <http://www.gnu.org/licenses/>.
//

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
