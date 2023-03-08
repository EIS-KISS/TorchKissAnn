#pragma once
#include <functional>
#include <filesystem>

torch::Tensor toeplitz(const torch::Tensor& a, const torch::Tensor b);

torch::Tensor fnToTensor(std::function<double(double)> fn, double start, double stop, size_t size);

double tensorSimpleIntegrate(torch::Tensor in, double length);

double integrateFn(std::function<double(double)> fn, double start, double stop, size_t steps = 100, double *prev = nullptr);

torch::Tensor unargmax(const torch::Tensor& in, int64_t size);

bool saveTensorToCsv(const std::filesystem::path& path, const torch::Tensor& tensor, const std::string& name);
