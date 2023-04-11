#pragma once
#include "torchph.h"

static constexpr int BATCH_SIZE = 100;
static constexpr auto DATA_LOCATION = torch::kCPU;
static constexpr int DATA_WIDTH = 100;
static constexpr int JOBS = 64;

typedef enum
{
	FILE_TYPE_INVALID = -1,
	FILE_TYPE_GENERATE = 0,
	FILE_TYPE_CSV,
	FILE_TYPE_MPT,
	FILE_TYPE_TRASH,
	FILE_TYPE_RELAXIS,
	FILE_TYPE_GAN
} FileType;
