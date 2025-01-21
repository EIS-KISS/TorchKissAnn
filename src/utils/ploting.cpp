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

#ifdef ENABLE_PLOTTING

bool save2dPlot(const std::filesystem::path& path, const std::string& xLabel, const std::string& yLabel,
	std::valarray<double> xData, std::valarray<double> yData, bool square, bool log, bool points)
{
	sciplot::Plot2D plot;
	plot.xlabel(xLabel);
	plot.ylabel(yLabel);

	double xMin = xData.min();
	double xMax = xData.max();
	double yMin = yData.min();
	double yMax = yData.max();

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

	if(points)
		plot.drawPoints(xData, yData);
	else
		plot.drawCurve(xData, yData);
	plot.legend().hide();

	sciplot::Figure fig({{plot}});
	sciplot::Canvas canvas({{fig}});
	canvas.size(640, 480);
	canvas.save(path);
	return true;
}

bool save2dPlot(const std::filesystem::path& path, const std::string& xLabel, const std::string& yLabel,
	std::vector<double> xData, std::vector<double> yData, bool square, bool log, bool points)
{
	return save2dPlot(path, xLabel, yLabel, std::valarray<double>(xData.data(), xData.size()), std::valarray<double>(yData.data(), yData.size()), square, log, points);
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
	return true;
}

#endif
