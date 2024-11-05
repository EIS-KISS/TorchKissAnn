#pragma once
#include <filesystem>
#include <string>
#include <valarray>
#include <vector>

bool save2dPlot(const std::filesystem::path& path, const std::string& xLabel, const std::string& yLabel,
                std::valarray<double> xData, std::valarray<double> yData, bool square = false, bool log = false, bool points = false);

bool save2dPlot(const std::filesystem::path& path, const std::string& xLabel, const std::string& yLabel,
                std::vector<double> xData, std::vector<double> yData, bool square = false, bool log = false, bool points = false);
