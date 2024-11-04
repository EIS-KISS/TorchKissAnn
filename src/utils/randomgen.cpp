#include "randomgen.h"
#include <assert.h>
#include <cstddef>
#include <cstdint>
#include <random>

static std::default_random_engine randomEngine;

double rd::rand(double min, double max)
{
	std::uniform_real_distribution<double> dist(min, max);
	return dist(randomEngine);
}

double rd::rand(double max)
{
	static std::uniform_real_distribution<double> dist(0, 1);
	return dist(randomEngine)*max;
}

size_t rd::uid()
{
	static std::uniform_int_distribution<size_t> distSt(0, SIZE_MAX);
	return distSt(randomEngine);
}

void rd::init()
{
	std::random_device randomDevice;
	randomEngine.seed(randomDevice());
}
