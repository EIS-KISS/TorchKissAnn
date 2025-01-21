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
#include "torchph.h"

/**
 * @brief Fits the given spectra to the given model
 *
 * @param spectra spectra to fit
 * @param omegas corrisponding omegas to the spectra given
 * @param modelString model to fit the specra to
 * @param startingParams optional starting parameters
 * @return a std::pair with first: fitted parameters and second: residual errors
 */
std::pair<torch::Tensor, torch::Tensor> eisFit(torch::Tensor spectra, torch::Tensor omegas, const std::string& modelString, torch::Tensor startingParams = torch::Tensor());
