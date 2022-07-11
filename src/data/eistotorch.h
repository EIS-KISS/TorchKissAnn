#pragma once

#include <eisgenerator/eistype.h>

#include <vector>

torch::Tensor eisToTorch(const std::vector<eis::DataPoint>& data, torch::Tensor* freqs = nullptr);
std::vector<eis::DataPoint> torchToEis(const torch::Tensor& input);
