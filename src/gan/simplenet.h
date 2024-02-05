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
