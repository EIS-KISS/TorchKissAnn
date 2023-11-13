#pragma once
#include <cstddef>
#include <limits>

class LossTermination
{
private:
	double prev = std::numeric_limits<double>::min();
	double nabla;
	size_t patienceCounter = 0;
	size_t patienceFactor;

public:
	LossTermination(size_t patienceFactorI = 20, double nablaI = 0.0001):
	nabla(nablaI), patienceFactor(patienceFactorI)
	{}


	bool terminate(double loss)
	{
		if((prev-loss) / prev < nabla )
			++patienceCounter;
		return patienceCounter > patienceFactor;
	}
};
