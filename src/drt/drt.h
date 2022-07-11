#pragma once
#include <eisgenerator/eistype.h>
#include <vector>

#include "../data/eistotorch.h"

void doDrt(const std::vector<eis::DataPoint>& data, torch::Tensor& drtOut);
