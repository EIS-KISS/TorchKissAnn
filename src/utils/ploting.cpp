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

#include "ploting.h"
#include <sciplot/Canvas.hpp>
#include <sciplot/Figure.hpp>
#include <sciplot/Plot2D.hpp>
#include <sciplot/Plot3D.hpp>

#ifdef ENABLE_PLOTTING

bool save2dPlot(const std::filesystem::path& path, const std::string& xLabel, const std::string& yLabel,
                std::vector<std::pair<std::valarray<double>, std::valarray<double>>> data, bool square, bool log, bool points)
{
		sciplot::Plot2D plot;
	plot.xlabel(xLabel);
	plot.ylabel(yLabel);

	std::valarray<double> xMaxC(data.size());
	std::valarray<double> xMinC(data.size());
	std::valarray<double> yMaxC(data.size());
	std::valarray<double> yMinC(data.size());

	for(size_t i = 0; i < data.size(); ++i)
	{
		xMaxC[i] = data[i].first.max();
		xMinC[i] = data[i].first.min();
		yMaxC[i] = data[i].second.max();
		yMinC[i] = data[i].second.min();
	}

	double xMin = xMinC.min();
	double xMax = xMaxC.max();
	double yMin = yMinC.min();
	double yMax = yMaxC.max();

	if(square)
	{
		xMin = std::min(xMin, yMin);
		yMin = xMin;

		xMax = std::max(xMax, yMax);
		yMax = xMax;
	}

	plot.xrange(xMin, xMax);
	plot.yrange(yMin, yMax);
	if(log)
		plot.ytics().logscale(10);

	for(const std::pair<std::valarray<double>, std::valarray<double>>& curve : data)
	{
		if(points)
			plot.drawPoints(curve.first, curve.second);
		else
			plot.drawCurve(curve.first, curve.second);
	}
	plot.legend().hide();

	sciplot::Figure fig({{plot}});
	sciplot::Canvas canvas({{fig}});
	canvas.size(640, 480);
	canvas.save(path);
	return true;
}


bool save2dPlot(const std::filesystem::path& path, const std::string& xLabel, const std::string& yLabel,
	std::valarray<double> xData, std::valarray<double> yData, bool square, bool log, bool points)
{
	return save2dPlot(path, xLabel, yLabel, {{xData, yData}}, square, log, points);
}

bool save2dPlot(const std::filesystem::path& path, const std::string& xLabel, const std::string& yLabel,
	std::vector<double> xData, std::vector<double> yData, bool square, bool log, bool points)
{
	return save2dPlot(path, xLabel, yLabel, std::valarray<double>(xData.data(), xData.size()), std::valarray<double>(yData.data(), yData.size()), square, log, points);
}

bool save3dPlot(const std::filesystem::path& path, const std::string& xLabel, const std::string& yLabel, const std::string& zLabel,
                std::valarray<double> xData, std::valarray<double> yData, std::valarray<double> zData, bool log)
{
	sciplot::Plot3D plot;
	plot.xlabel(xLabel);
	plot.ylabel(yLabel);
	plot.zlabel(zLabel);

	double xMin = xData.min();
	double xMax = xData.max();
	double yMin = yData.min();
	double yMax = yData.max();
	double zMin = zData.min();
	double zMax = zData.max();

	plot.xrange(xMin, xMax);
	plot.yrange(yMin, yMax);
	plot.zrange(zMin, zMax);

	if(log)
	{
		plot.xtics().logscale(10);
		plot.ytics().logscale(10);
		plot.ztics().logscale(10);
	}

	plot.drawPoints(xData, yData, zData);


	sciplot::Figure fig({{plot}});
	sciplot::Canvas canvas({{fig}});
	canvas.size(1280, 960);

	canvas.save(path);

	return true;
}

bool save3dPlot(const std::filesystem::path& path, const std::string& xLabel, const std::string& yLabel, const std::string& zLabel,
                std::vector<double> xData, std::vector<double> yData, std::vector<double> zData, bool log)
{
	return save3dPlot(path, xLabel, yLabel, zLabel, std::valarray<double>(xData.data(), xData.size()), std::valarray<double>(yData.data(), yData.size()), std::valarray<double>(zData.data(), zData.size()), log);
}

#else

bool save2dPlot(const std::filesystem::path& path, const std::string& xLabel, const std::string& yLabel,
	std::valarray<double> xData, std::valarray<double> yData, bool square, bool log, bool points)
{
	(void)path;
	(void)yLabel;
	(void)xLabel;
	(void)xData;
	(void)yData;
	(void)square;
	(void)log;
	(void)points;
	return true;
}


bool save2dPlot(const std::filesystem::path& path, const std::string& xLabel, const std::string& yLabel,
	std::vector<double> xData, std::vector<double> yData, bool square, bool log, bool points)
{
	return save2dPlot(path, xLabel, yLabel, std::valarray<double>(xData.data(), xData.size()), std::valarray<double>(yData.data(), yData.size()), square, log, points);
}

bool save3dPlot(const std::filesystem::path& path, const std::string& xLabel, const std::string& yLabel, const std::string& zLabel,
                std::valarray<double> xData, std::valarray<double> yData, std::valarray<double> zData, bool log = false)
{
	(void)path;
	(void)yLabel;
	(void)xLabel;
	(void)zLabel;
	(void)xData;
	(void)yData;
	(void)zData;
	(void)log;
	return true;
}

bool save3dPlot(const std::filesystem::path& path, const std::string& xLabel, const std::string& yLabel, const std::string& zLabel,
                std::vector<double> xData, std::vector<double> yData, std::vector<double> zData, bool log = false)
{
	return save3dPlot(path, xLabel, yLabel, zLabel, std::valarray<double>(xData.data(), xData.size()), std::valarray<double>(yData.data(), yData.size()), std::valarray<double>(zData.data(), zData.size()), log);
}

#endif
