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
