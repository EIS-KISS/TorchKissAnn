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

#include "torchph.h"
#include "net.h"

struct GANGeneratorImpl : public ann::Net
{
private:
    torch::nn::Sequential model;

    void init();

public:
    //feature: the number of filters in convolution layer closest to image
    //latSize: dimensions of latent space
    GANGeneratorImpl(const Json::Value& node);
    GANGeneratorImpl(size_t inputSize = 10, size_t outputSize = 100);
    virtual torch::Tensor forward(torch::Tensor z) override;
    virtual std::shared_ptr<torch::nn::Module> operator[](size_t index) override;
    virtual bool saveToCheckpointDir(const std::filesystem::path& path) override;
};

struct GANDiscriminatorImpl : public ann::Net
{
private:
    torch::nn::Sequential feature;
    torch::nn::Sequential classifier;

    void init();

public:
    GANDiscriminatorImpl(const Json::Value& node);
    GANDiscriminatorImpl(size_t inputSize = 100);
    std::pair<torch::Tensor, torch::Tensor> forwardSplit(torch::Tensor x);
    virtual torch::Tensor forward(torch::Tensor x) override;
    virtual std::shared_ptr<torch::nn::Module> operator[](size_t index) override;
    virtual bool saveToCheckpointDir(const std::filesystem::path& path) override;
};

TORCH_MODULE(GANGenerator);
TORCH_MODULE(GANDiscriminator);
