import torch

@torch.jit.script
def cap(omega, c):
	cmplx = torch.ones(omega.shape, dtype=torch.complex64, device=omega.device)*1j
	cmplx = (cmplx/(omega*c))*-1
	return cmplx

@torch.jit.script
def resistor(omega, r):
	return torch.ones(omega.shape, dtype=torch.complex64, device=omega.device)*r

@torch.jit.script
def parallel(a, b):
	return 1/(1/a+1/b)

@torch.jit.script
def serial(a, b):
	return a+b


@torch.jit.script
def forward(omega, parameters):
	capTensor = cap(omega, parameters[1])
	reistorTensor = resistor(omega, parameters[0])
	parallelTensor = parallel(capTensor, reistorTensor)

	capTensorTwo = cap(omega, parameters[3])
	reistorTensorTwo = resistor(omega, parameters[2])
	parallelTensorTwo = parallel(capTensorTwo, reistorTensorTwo)

	result = serial(parallelTensor, parallelTensorTwo)

	return torch.cat((result.real, result.imag), 0)


def getParametersForStep(parameterRanges, step, stepPerRange, device):
	steps = []
	for i in reversed(range(0, parameterRanges.size(0))):
		magnitude = pow(stepPerRange, i)
		steps.append(int(step/magnitude))
		step = step % magnitude
		print(f'{magnitude}, {step}, {steps[-1]}')

	parameters = torch.empty((parameterRanges.size(0)))
	for i in range(0, parameters.numel()):
		delta = abs(parameterRanges[i][1]-parameterRanges[i][0])/stepPerRange
		parameters[i] = min(parameterRanges[i][1], parameterRanges[i][0])+delta*steps[i]
	parameters.to(device)
	return parameters

if __name__ == "__main__":
	cpuDevice = torch.device("cpu")
	if torch.cuda.is_available():
		print("using GPU")
		offloadDevice = torch.device("cuda:0")
	else:
		print("using CPU")
		offloadDevice = cpuDevice

	omega = (torch.logspace(0, torch.log10(torch.tensor(1e6)), 50, 10, device=offloadDevice)*1)
	parameterRanges = torch.tensor([[10, 100],[1e-4, 1e-6], [10, 100], [1e-6, 1e-9]])
	parameters = getParametersForStep(parameterRanges, 100, 10, offloadDevice)
	print(parameters)
	result  = forward(omega, parameters)
	print(omega.to(cpuDevice))
	print(result.to(cpuDevice))

	torch.jit.save(forward, "testScript.ts")
