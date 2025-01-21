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
