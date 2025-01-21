/* * TorchKissAnn - A collection of tools to train various types of Machine learning
 * algorithms on various types of EIS data
 * Copyright (C) 2025 Carl Klemm <carl@uvos.xyz>
 *
 * This file is part of TorchKissAnn.
 *
 * TorchKissAnn is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * TorchKissAnn is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with TorchKissAnn.  If not, see <http://www.gnu.org/licenses/>.
 */

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
