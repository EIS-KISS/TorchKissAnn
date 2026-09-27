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
#include <filesystem>
#include <string>
#include <valarray>
#include <vector>

enum PlotType
{
    PLOT_TYPE_SCATTER,
    PLOT_TYPE_LINE
};

struct PlotOptions
{
    std::string xLabel = "X Axis";
    std::string yLabel = "Y Axis";
    std::string zLabel = "Z Axis";
    bool square = false;
    bool xlog = false;
    bool ylog = false;
    bool zlog = false;
    enum PlotType type = PLOT_TYPE_LINE;
};

bool save2dPlot(const std::filesystem::path& path, std::valarray<double> xData, std::valarray<double> yData, const PlotOptions& options);

bool save2dPlot(const std::filesystem::path& path, std::vector<std::pair<std::valarray<double>, std::valarray<double>>> data, const PlotOptions& options);

bool save2dPlot(const std::filesystem::path& path, std::vector<double> xData, std::vector<double> yData, const PlotOptions& options);

bool save3dPlot(const std::filesystem::path& path, std::valarray<double> xData, std::valarray<double> yData, std::valarray<double> zData, const PlotOptions& options);

bool save3dPlot(const std::filesystem::path& path, std::vector<double> xData, std::vector<double> yData, std::vector<double> zData, const PlotOptions& options);
