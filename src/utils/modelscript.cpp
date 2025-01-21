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

#include "modelscript.h"

#include "data/eistotorch.h"

std::shared_ptr<torch::CompilationUnit> compileModel(eis::Model &model)
{
	std::string torchScript = model.getTorchScript();
	std::shared_ptr<torch::CompilationUnit> compiledModule = torch::jit::compile(torchScript);
	return compiledModule;
}

std::shared_ptr<torch::CompilationUnit> compileModel(std::string modelstr)
{
	eis::Model model(modelstr);
	return compileModel(model);
}

torch::Tensor runScriptModel(eis::Model& model, std::shared_ptr<torch::CompilationUnit> compiledScript,
							torch::Tensor parameters, torch::Tensor omegas)
{
	torch::Tensor result = compiledScript->run_method(model.getCompiledFunctionName(), parameters, omegas).toTensor();
	assert(result.scalar_type() == torch::kComplexFloat || result.scalar_type() == torch::kComplexDouble);
	assert(result.size(0) == omegas.numel());
	assert(result.sizes().size() == 1 || result.size(1) == 1);
	if(result.sizes().size() == 2)
		result = result.reshape({result.numel()});
	else
		assert(false);
	return result;
}

torch::Tensor runScriptModel(eis::Model& model, size_t step, std::shared_ptr<torch::CompilationUnit> compiledScript, torch::Tensor omegas)
{
	model.resolveSteps(step);
	std::vector<fvalue> parameterVec = model.getFlatParameters();
	torch::Tensor parameters = fvalueVectorToTensor(parameterVec);
	return runScriptModel(model, compiledScript, parameters, omegas);
}
