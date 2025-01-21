/* * TorchKissAnn - A collection of tools to train various types of Machine learning
 * algorithms on various types of EIS data
 * Copyright (C) 2025 Carl Klemm <carl@uvos.xyz>
 *
 * This file is part of TorchKissAnn.
 *
 * TorchKissAnn is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * TorchKissAnn is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with TorchKissAnn.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <cstddef>
#include <utility>

// Define Namespace
namespace nn = torch::nn;

// Function Prototype
void weightsInit(nn::Module &m);
void downSampling(nn::Sequential &sq, const size_t in_nc, const size_t out_nc, const bool BN, const bool LReLU, const bool bias = false);
void upSampling(nn::Sequential &sq, const size_t in_nc, const size_t out_nc, const bool BN, const bool ReLU, const bool bias = false);


struct GANGeneratorConvImpl : public nn::Module
{
private:
    nn::Sequential model;
public:
    //feature: the number of filters in convolution layer closest to image
    //latSize: dimensions of latent space
    GANGeneratorConvImpl(size_t feature = 64, size_t inputSize = 100, size_t latSize = 512);
    torch::Tensor forward(torch::Tensor z);
};

struct GANDiscriminatorConvImpl : public nn::Module
{
private:
    nn::Sequential down;
    nn::Sequential features;
    nn::Sequential classifier;
public:
    GANDiscriminatorConvImpl(size_t feature = 64, size_t depth = 100);
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x);
};

TORCH_MODULE(GANGeneratorConv);
TORCH_MODULE(GANDiscriminatorConv);
