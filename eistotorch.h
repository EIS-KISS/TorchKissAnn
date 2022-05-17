#pragma once

#include <eisgenerator/eistype.h>

torch::Tensor eisToTorch(const std::vector<eis::DataPoint>& data);
