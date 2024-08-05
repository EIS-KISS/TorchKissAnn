#pragma once
#include <filesystem>
#include <string>
#include <valarray>

bool save2dPlot(const std::filesystem::path& path, const std::string& xLabel, const std::string& yLabel,
                std::valarray<double> xData, std::valarray<double> yData, bool square = false, bool log = false);
