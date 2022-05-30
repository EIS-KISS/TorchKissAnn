#pragma once

#include <eisgenerator/eistype.h>

torch::Tensor eisToTorch(const std::vector<eis::DataPoint>& data);
std::vector<eis::DataPoint> torchToEis(const torch::Tensor& input);
