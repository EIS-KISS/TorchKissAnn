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

#ifdef ENABLE_PLOTTING

#include <sciplot/Canvas.hpp>
#include <sciplot/Figure.hpp>
#include <sciplot/Plot2D.hpp>
#include <sciplot/Plot3D.hpp>

bool save2dPlot(const std::filesystem::path& path, std::vector<std::pair<std::valarray<double>, std::valarray<double>>> data, const PlotOptions& options)
{
	sciplot::Plot2D plot;
	plot.xlabel(options.xLabel);
	plot.ylabel(options.yLabel);

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

	if(options.square)
	{
		xMin = std::min(xMin, yMin);
		yMin = xMin;

		xMax = std::max(xMax, yMax);
		yMax = xMax;
	}

	plot.xrange(xMin, xMax);
	plot.yrange(yMin, yMax);
	if(options.xlog)
		plot.xtics().logscale(10);
	if(options.ylog)
		plot.ytics().logscale(10);

	for(const std::pair<std::valarray<double>, std::valarray<double>>& curve : data)
	{
		if(options.type == PLOT_TYPE_SCATTER)
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


bool save2dPlot(const std::filesystem::path& path, std::valarray<double> xData, std::valarray<double> yData, const PlotOptions& options)
{
	return save2dPlot(path, {{xData, yData}}, options);
}

bool save2dPlot(const std::filesystem::path& path, std::vector<double> xData, std::vector<double> yData, const PlotOptions& options)
{
	return save2dPlot(path, std::valarray<double>(xData.data(), xData.size()), std::valarray<double>(yData.data(), yData.size()), options);
}

bool save3dPlot(const std::filesystem::path& path, std::valarray<double> xData, std::valarray<double> yData,
                std::valarray<double> zData, const PlotOptions& options)
{
	sciplot::Plot3D plot;
	plot.xlabel(options.xLabel);
	plot.ylabel(options.yLabel);
	plot.zlabel(options.zLabel);

	double xMin = xData.min();
	double xMax = xData.max();
	double yMin = yData.min();
	double yMax = yData.max();
	double zMin = zData.min();
	double zMax = zData.max();

	plot.xrange(xMin, xMax);
	plot.yrange(yMin, yMax);
	plot.zrange(zMin, zMax);

	if(options.xlog)
		plot.xtics().logscale(10);
	if(options.ylog)
		plot.ytics().logscale(10);
	if(options.zlog)
		plot.ztics().logscale(10);

	plot.drawPoints(xData, yData, zData);


	sciplot::Figure fig({{plot}});
	sciplot::Canvas canvas({{fig}});
	canvas.size(1280, 960);

	canvas.save(path);

	return true;
}

bool save3dPlot(const std::filesystem::path& path,
                std::vector<double> xData, std::vector<double> yData, std::vector<double> zData, const PlotOptions& options)
{
	return save3dPlot(path,
					  std::valarray<double>(xData.data(), xData.size()),
					  std::valarray<double>(yData.data(), yData.size()),
					  std::valarray<double>(zData.data(), zData.size()),
					  options);
}

#else

bool save2dPlot(const std::filesystem::path& path, std::vector<std::pair<std::valarray<double>, std::valarray<double>>> data, const PlotOptions& options)
{
	return true;
}


bool save2dPlot(const std::filesystem::path& path, std::valarray<double> xData, std::valarray<double> yData, const PlotOptions& options)
{
	return save2dPlot(path, {{xData, yData}}, options);
}

bool save2dPlot(const std::filesystem::path& path, std::vector<double> xData, std::vector<double> yData, const PlotOptions& options)
{
	return save2dPlot(path, std::valarray<double>(xData.data(), xData.size()), std::valarray<double>(yData.data(), yData.size()), options);
}

bool save3dPlot(const std::filesystem::path& path, std::valarray<double> xData, std::valarray<double> yData,
                std::valarray<double> zData, const PlotOptions& options)
{
	return true;
}

bool save3dPlot(const std::filesystem::path& path,
                std::vector<double> xData, std::vector<double> yData, std::vector<double> zData, const PlotOptions& options
{
	return save3dPlot(path,
					  std::valarray<double>(xData.data(), xData.size()),
					  std::valarray<double>(yData.data(), yData.size()),
					  std::valarray<double>(zData.data(), zData.size()),
					  options);
}

#endif
