#pragma once
#include <eisgenerator/eistype.h>
#include <vector>
#include <string>

namespace loaders
{
	std::vector<eis::DataPoint> csv(std::string fileName);
}
