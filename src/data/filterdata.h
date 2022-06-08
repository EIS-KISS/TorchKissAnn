#pragma once

#include <eisgenerator/normalize.h>
#include <eisgenerator/basicmath.h>

inline void filterData(std::vector<eis::DataPoint>& data, size_t outputSize)
{
	data = eis::reduceRegion(data);

	if(data.size() < outputSize/8)
	{
		data = std::vector<eis::DataPoint>();
		return;
	}
	data = eis::rescale(data, outputSize/2);
}
