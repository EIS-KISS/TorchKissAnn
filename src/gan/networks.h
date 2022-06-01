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
