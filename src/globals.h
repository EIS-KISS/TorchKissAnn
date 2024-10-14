#pragma once
#include "torchph.h"

static constexpr int DATA_WIDTH = 100;
static constexpr int JOBS = 64;

extern torch::DeviceType offload_type;
extern torch::Device* offload_device;
extern int batch_size;

typedef enum
{
	FILE_TYPE_INVALID = -1,
	FILE_TYPE_GENERATE = 0,
	FILE_TYPE_CSV,
	FILE_TYPE_TRASH,
	FILE_TYPE_RELAXIS,
	FILE_TYPE_GAN
} FileType;

void choose_device(bool forceCpu = false);
void free_device();
