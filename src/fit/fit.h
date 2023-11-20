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
