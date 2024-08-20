#pragma once

#include "torchph.h"

// Dim 0 features, Dim 1 examples
torch::Tensor  r2score(const torch::Tensor& prediction, const torch::Tensor& truth);
