#include "randomgen.h"
#include <assert.h>
#include <cstddef>
#include <cstdint>
#include <random>

static std::default_random_engine randomEngine;
static std::uniform_real_distribution<double> dist(0, 1);
static std::uniform_int_distribution<size_t> distSt(0, SIZE_MAX);

double rd::rand(double max)
{
	return dist(randomEngine)*max;
}

size_t rd::uid()
{
	return distSt(randomEngine);
}

void rd::init()
{
	std::random_device randomDevice;
	randomEngine.seed(randomDevice());
}
