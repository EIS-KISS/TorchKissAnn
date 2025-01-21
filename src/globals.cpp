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

#include <c10/core/DeviceType.h>
#include <cstddef>
#include <torch/cuda.h>
#include <torch/torch.h>
#include <limits>

#if (defined(CUDA_VERSION) && CUDA_VERSION >= 11000) || defined(USE_ROCM)
#include <ATen/hip/HIPContext.h>
#elif defined(CUDA_VERSION)
#include <ATen/cuda/CUDAContext.h>
#endif


#include "log.h"

torch::DeviceType offload_type;
torch::Device* offload_device;
int batch_size = 256;

static size_t print_device_proparties(size_t deviceIndex, bool newline = true)
{
#if (defined(CUDA_VERSION) && CUDA_VERSION >= 11000) || defined(USE_ROCM)
	hipDeviceProp_t* deviceProparties = at::cuda::getDeviceProperties(deviceIndex);
	if(!deviceProparties)
	{
		Log(Log::ERROR, newline)<<"Could not get proparties for device "<<deviceIndex;
		return 0;
	}
	Log(Log::INFO, newline)<<"Hip device "<<deviceIndex<<" \""<<deviceProparties->name<<"\" with "
		<<deviceProparties->totalGlobalMem/(1024*1024)<<"MiB vram, architecture: "
		<<deviceProparties->gcnArchName;
	return deviceProparties->totalGlobalMem;
#elif defined(CUDA_VERSION)
		cudaDeviceProp* deviceProparties = at::cuda::getDeviceProperties(i);
		Log(Log::INFO, newline)<<"Cuda device "<<deviceIndex<<" \""<<deviceProparties->name<<"\" with "
			<<deviceProparties->totalGlobalMem/(1024*1024)<<"MiB vram";
	return deviceProparties->totalGlobalMem;
#endif
	return 0;
}

void choose_device(bool forceCpu)
{
	if(torch::cuda::is_available() && !forceCpu)
	{
		Log(Log::INFO)<<"Using GPU";
		offload_type = torch::kCUDA;

		Log(Log::INFO)<<"Found "<<torch::cuda::device_count()<<" GPUs";
		size_t maxMemory = 0;
		size_t maxMemoryIndex = 0;
		for(size_t i = 0; i < torch::cuda::device_count(); ++i)
		{
			size_t memory = print_device_proparties(i);
			if(memory > maxMemory)
			{
				maxMemory = memory;
				maxMemoryIndex = i;
			}
		}

		Log(Log::INFO, false)<<"Will use ";
		print_device_proparties(maxMemoryIndex, false);
		Log(Log::INFO)<<" because this device has the most memory\n"
			<<"if this is not desirable please set the HIP_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES envvar";

		offload_device = new torch::Device(offload_type, maxMemoryIndex);
	}
	else
	{
		Log(Log::INFO)<<"Using CPU";
		offload_type = torch::kCPU;
		offload_device = new torch::Device(offload_type, 0);
	}
}

void free_device()
{
	if(offload_device)
		delete offload_device;
}
