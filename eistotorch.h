#pragma once

#include <torch/torch.h>
#include <eisgenerator/eistype.h>

torch::Tensor eisToTorch(const std::vector<eis::DataPoint>& data);
