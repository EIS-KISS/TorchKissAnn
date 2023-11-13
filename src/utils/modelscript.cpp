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
