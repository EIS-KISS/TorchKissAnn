#pragma once
#include <cstddef>

struct GANGeneratorImpl : public torch::nn::Module
{
private:
    torch::nn::Sequential model;
public:
    //feature: the number of filters in convolution layer closest to image
    //latSize: dimensions of latent space
    GANGeneratorImpl(size_t inputSize = 10, size_t outputSize = 100);
    torch::Tensor forward(torch::Tensor z);
};

struct GANDiscriminatorImpl : public torch::nn::Module
{
private:
    torch::nn::Sequential classifier;
public:
    GANDiscriminatorImpl(size_t inputSize = 100);
    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(GANGenerator);
TORCH_MODULE(GANDiscriminator);
