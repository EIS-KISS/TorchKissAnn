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

#include <cstddef>
#include <utility>
#include <typeinfo>
#include <cmath>
#include "networks.h"

namespace nn = torch::nn;

GANGeneratorConvImpl::GANGeneratorConvImpl(size_t feature, size_t inputSize, size_t latSize)
{
    upSampling(model, latSize, feature*8, /*BN=*/true, /*ReLU=*/true);  // {Z,1,1}     ===> {8F,2,2}
    for (size_t i = 0; i < inputSize - 5; i++)
        upSampling(model, feature*8, feature*8, /*BN=*/true, /*ReLU=*/true);          // {8F,2,2}    ===> {8F,16,16}
    upSampling(model, feature*8, feature*4, /*BN=*/true, /*ReLU=*/true);              // {8F,16,16}  ===> {4F,32,32}
    upSampling(model, feature*4, feature*2, /*BN=*/true, /*ReLU=*/true);              // {4F,32,32}  ===> {2F,64,64}
    upSampling(model, feature*2, feature, /*BN=*/true, /*ReLU=*/true);                // {2F,64,64}  ===> {F,128,128}
    upSampling(model, feature, 1, /*BN=*/false, /*ReLU=*/false);  // {F,128,128} ===> {C,256,256}
    model->push_back(nn::Tanh());                                                     // [-inf,+inf] ===> [-1,1]
    register_module("Generator", model);
}


torch::Tensor GANGeneratorConvImpl::forward(torch::Tensor z)
{
    z = z.view({z.size(0), z.size(1), 1, 1});     // {Z} ===> {Z,1,1}
    torch::Tensor out = model->forward(z);  // {Z,1,1} ===> {C,256,256}
    return out;
}

GANDiscriminatorConvImpl::GANDiscriminatorConvImpl(size_t feature, size_t inputSize)
{
    downSampling(down, 1, feature, /*BN=*/false, /*LReLU=*/true);  // {C,256,256} ===> {F,128,128}
    downSampling(down, feature, feature*2, /*BN=*/true, /*LReLU=*/true);               // {F,128,128} ===> {2F,64,64}
    downSampling(down, feature*2, feature*4, /*BN=*/true, /*LReLU=*/true);             // {2F,64,64}  ===> {4F,32,32}
    downSampling(down, feature*4, feature*8, /*BN=*/true, /*LReLU=*/true);             // {4F,32,32}  ===> {8F,16,16}
    for (size_t i = 0; i < inputSize - 5; i++)
        downSampling(down, feature*8, feature*8, /*BN=*/true, /*LReLU=*/true);         // {8F,16,16}  ===> {8F,2,2}
    register_module("down", down);

    features = nn::Sequential(
        nn::Linear(feature*8*2*2, feature*16),  // {32F} ===> {16F}
        nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2).inplace(true))
    );
    register_module("features", features);

    classifier = nn::Sequential(
        nn::Linear(feature*16, 1)  // {16F} ===> {1}
    );
    register_module("classifier", classifier);
}

std::pair<torch::Tensor, torch::Tensor> GANDiscriminatorConvImpl::forward(torch::Tensor x)
{
    torch::Tensor mid, feature, out;
    std::pair<torch::Tensor, torch::Tensor> out_with_feature;
    mid = down->forward(x);              // {C,256,256} ===> {8F,2,2}
    mid = mid.view({mid.size(0), -1});         // {8F,2,2}    ===> {32F}
    feature = features->forward(mid);    // {32F}       ===> {16F}
    out = classifier->forward(feature);  // {16F}       ===> {1}
    out_with_feature = {out, feature};
    return out_with_feature;
}

void weightsInit(nn::Module &m)
{
    if ((typeid(m) == typeid(nn::Conv2d)) || (typeid(m) == typeid(nn::Conv2dImpl)) ||
        (typeid(m) == typeid(nn::ConvTranspose2d)) || (typeid(m) == typeid(nn::ConvTranspose2dImpl)))
    {
        auto p = m.named_parameters(false);
        auto w = p.find("weight");
        auto b = p.find("bias");
        if (w != nullptr)
            nn::init::normal_(*w, /*mean=*/0.0, /*std=*/0.02);
        if (b != nullptr)
            nn::init::constant_(*b, /*bias=*/0.0);
    }
    else if ((typeid(m) == typeid(nn::Linear)) || (typeid(m) == typeid(nn::LinearImpl)))
    {
        auto p = m.named_parameters(false);
        auto w = p.find("weight");
        auto b = p.find("bias");
        if (w != nullptr)
            nn::init::normal_(*w, /*mean=*/0.0, /*std=*/0.02);
        if (b != nullptr)
            nn::init::constant_(*b, /*bias=*/0.0);
    }
    else if ((typeid(m) == typeid(nn::BatchNorm2d)) || (typeid(m) == typeid(nn::BatchNorm2dImpl)))
    {
        auto p = m.named_parameters(false);
        auto w = p.find("weight");
        auto b = p.find("bias");
        if (w != nullptr)
            nn::init::normal_(*w, /*mean=*/1.0, /*std=*/0.02);
        if (b != nullptr)
            nn::init::constant_(*b, /*bias=*/0.0);
    }
}

void downSampling(nn::Sequential &sq, const size_t in_nc, const size_t out_nc, const bool BN, const bool LReLU, const bool bias)
{
    sq->push_back(nn::Conv2d(nn::Conv2dOptions(in_nc, out_nc, 4).stride(2).padding(1).bias(bias)));
    if (BN)
        sq->push_back(nn::BatchNorm2d(out_nc));
    if (LReLU)
        sq->push_back(nn::LeakyReLU(nn::LeakyReLUOptions().negative_slope(0.2).inplace(true)));
}

void upSampling(nn::Sequential &sq, const size_t in_nc, const size_t out_nc, const bool BN, const bool ReLU, const bool bias)
{
    sq->push_back(nn::ConvTranspose2d(nn::ConvTranspose2dOptions(in_nc, out_nc, 4).stride(2).padding(1).bias(bias)));
    if (BN)
        sq->push_back(nn::BatchNorm2d(out_nc));
    if (ReLU)
        sq->push_back(nn::ReLU(nn::ReLUOptions().inplace(true)));
}
