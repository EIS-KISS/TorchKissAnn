import torch

class Model(torch.nn.Module):
	def __init__(self):
		super(Model, self).__init__()

	@staticmethod
	def cap(omega, c):
		cmplx = torch.ones(omega.shape, dtype=torch.complex64, device=omega.device)*1j
		cmplx = (cmplx/(omega*c))*-1
		return cmplx

	@staticmethod
	def resistor(omega, r):
		return torch.ones(omega.shape, dtype=torch.complex64, device=omega.device)*r

	@staticmethod
	def parallel(a, b):
		return 1/(1/a+1/b)

	@staticmethod
	def serial(a, b):
		return a+b

	@torch.jit.export
	def forward(self, omega, parameters):
		capTensor = self.cap(omega, parameters[1])
		reistorTensor = self.resistor(omega, parameters[0])
		parallel = self.parallel(capTensor, reistorTensor)

		capTensorTwo = self.cap(omega, parameters[3])
		reistorTensorTwo = self.resistor(omega, parameters[2])
		parallelTwo = self.parallel(capTensorTwo, reistorTensorTwo)

		result = self.serial(parallel, parallelTwo)

		return torch.cat((result.real, result.imag), 0)

	@torch.jit.export
	def paramerterCount():
		return 4

if __name__ == "__main__":
	if torch.cuda.is_available():
		print("using GPU")
		device = torch.device("cuda:0")
	else:
		print("using CPU")
		device = torch.device("cpu")

	model = Model()
	omega = (torch.logspace(0, torch.log10(torch.tensor(1e6)), 3, 10)*1).to(device)
	parameters = torch.empty((4))
	parameters[0] = 100
	parameters[1] = 1e-4
	parameters[2] = 100
	parameters[3] = 1e-6
	parameters = parameters.to(device)
	result  = model.forward(omega, parameters)
	print(omega)
	print(result)
