//
// TorchKissAnn - A collection of tools to train various types of Machine learning
// algorithms on various types of EIS data
// Copyright (C) 2025 Carl Klemm <carl@uvos.xyz>
//
// This file is part of TorchKissAnn.
//
// TorchKissAnn is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// TorchKissAnn is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with TorchKissAnn.  If not, see <http://www.gnu.org/licenses/>.
//

#include <eisgenerator/log.h>

#include "ann/classification.h"
#include "ann/regression.h"
#include "gan/gan.h"
#include "log.h"
#include "options.h"
#include "globals.h"

int main(int argc, char** argv)
{
	std::cout<<std::setprecision(5)<<std::fixed<<std::setw(3);
	Log::level = Log::INFO;
	eis::Log::level = eis::Log::ERROR;

	Config config;
	argp_parse(&argp, argc, argv, 0, 0, &config);

	choose_device(config.noGpu);

	if(config.mode == MODE_INVALID)
	{
		Log(Log::ERROR)<<"You must specify what to train: -m " MODE_LIST;
		return 3;
	}

	switch (config.mode)
	{
		case MODE_ANN:
		case MODE_ANN_CONV:
		case MODE_GAN:
			Log(Log::ERROR)<<"Tuneing is not yet supported in the mode "<<trainModeToStr(config.mode);
			break;

		case MODE_REGRESSION:
			//ann::regression::parameterSearch();
			//break;
		default:
			Log(Log::ERROR)<<"You must specify what to train: -m " MODE_LIST;
			return 3;
	}

	free_device();
	return 0;
}
