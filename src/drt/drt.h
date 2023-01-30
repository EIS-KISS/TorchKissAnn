#pragma once
#include <eisgenerator/eistype.h>

#ifdef DRT_ENABLED
std::vector<fvalue> calcDrt(const std::vector<eis::DataPoint>& data);
#endif
