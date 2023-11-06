#pragma once

#include <eisgenerator/model.h>
#include <string>
#include <memory>
#include <torch/jit.h>

#include "torchph.h"

std::shared_ptr<torch::CompilationUnit> compileModel(eis::Model &model);
std::shared_ptr<torch::CompilationUnit> compileModel(std::string modelstr);

torch::Tensor runScriptModel(eis::Model& model, std::shared_ptr<torch::CompilationUnit> compiledScript,
							torch::Tensor parameters, torch::Tensor omegas);

torch::Tensor runScriptModel(eis::Model& model, size_t step,
							std::shared_ptr<torch::CompilationUnit> compiledScript,
							torch::Tensor omegas);
