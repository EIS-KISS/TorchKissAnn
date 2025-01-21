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

#include <ATen/TensorIndexing.h>
#include <kisstype/type.h>
#include <filesystem>
#include <iostream>
#include <sciplot/Canvas.hpp>
#include <sciplot/Figure.hpp>
#include <sciplot/Plot2D.hpp>
#include <sciplot/Vec.hpp>
#include <sstream>
#include <fstream>
#include <sciplot/sciplot.hpp>
#include <string>
#include <torch/types.h>
#include <limits>

#include "commonoptions.h"
#include "data/loaders/regressionloader.h"
#include "data/loaders/regressiondirloader.h"
#include "data/loaders/tarloader.h"
#include "log.h"
#include "data/loaders/dirloader.h"
#include "data/eistotorch.h"
#include "data/classextractordataset.h"
#include "options.h"
#include "randomgen.h"
#include "tensoroperators.h"
#include "indicators.hpp"

template <typename DataSetType>
std::string generateReport(EisDataset<DataSetType>* dataset)
{
	std::stringstream ss;

	ss<<"Name,\tSize,\tClass Count,\tMulitclass\n";
	ss<<typeid(DataSetType).name()<<",\t"<<dataset->size().value()
	  <<",\t"<<dataset->outputSize()<<",\t"<<dataset->isMulticlass()<<'\n';

	for(size_t i = 0; i < dataset->outputSize(); ++i)
	{
		ss<<i;
		if(i+1 < dataset->outputSize())
			ss<<",\t";
	}
	ss<<'\n';
	for(size_t i = 0; i < dataset->outputSize(); ++i)
	{
		ss<<dataset->outputName(i);
		if(i+1 < dataset->outputSize())
			ss<<",\t";
	}
	ss<<'\n';
	ss<<tensorToString(dataset->classCounts())<<'\n';
	return ss.str();
}

template <typename DataSetType>
bool saveImages(EisDataset<DataSetType>* dataset, const std::filesystem::path& outDir)
{
	if(!std::filesystem::is_directory(outDir))
	{
		if(!std::filesystem::create_directory(outDir))
			return false;
	}
	size_t outputSize = dataset->outputSize();

	for(size_t i = 0; i < outputSize; ++i)
	{
		Log(Log::INFO)<<"Processing class "<<i+1<<" of "<<outputSize;
		ClassExtractorDataset<DataSetType> extractedClass(dataset, i);

		std::set<size_t> ids;

		if(extractedClass.size().value() < 1000)
		{
			Log(Log::INFO)<<"Class has "<<extractedClass.size().value()<<" examples";
			for(size_t i = 0; i < extractedClass.size().value(); ++i)
				ids.insert(i);
		}
		else
		{
			Log(Log::INFO)<<"Class has "<<extractedClass.size().value()<<" examples; will decimate to 500";
			while(ids.size() < 500)
				ids.insert(rd::rand(extractedClass.size().value()));
		}

		if(ids.empty())
		{
			Log(Log::ERROR)<<"Dataset dose not have examples every class it claims to have!!!";
			return false;
		}

		sciplot::Plot2D plot;
		plot.xlabel("Re");
		plot.ylabel("Im");

		double xmin = std::numeric_limits<double>::max();
		double xmax = std::numeric_limits<double>::min();
		double ymin = std::numeric_limits<double>::max();
		double ymax = std::numeric_limits<double>::min();

		indicators::BlockProgressBar bar(
			indicators::option::BarWidth(50),
			indicators::option::PrefixText("Processing spectra: "),
			indicators::option::ShowElapsedTime(true),
			indicators::option::ShowRemainingTime(true),
			indicators::option::MaxProgress(ids.size())
		);

		for(auto j : ids)
		{
			bar.tick();
			torch::data::Example<torch::Tensor, torch::Tensor> example = extractedClass.get(j);
			std::pair<std::valarray<double>, std::valarray<double>> data = torchToValarray<double>(example.data.to(torch::kFloat64));


			double cxmin = data.first.min();
			cxmin = cxmin - std::abs(cxmin*0.05);
			double cymin = data.second.min();
			cymin = cymin - std::abs(cymin*0.05);

			double cxmax = data.first.max();
			cxmax = cxmax - std::abs(cxmax*0.05);
			double cymax = data.second.max();
			cymax = cymax - std::abs(cymax*0.05);

			if(xmax < cxmax)
				xmax = cxmax;
			if(ymax < cymax)
				ymax = cymax;

			if(xmin > cxmin)
				xmin = cxmin;
			if(ymin > cymin)
				ymin = cymin;
			plot.xrange(xmin, xmax);
			plot.yrange(ymax, ymin);
			plot.legend().hide();

			plot.drawCurve(data.first, data.second);
		}
		sciplot::Figure fig({{plot}});
		sciplot::Canvas canvas({{fig}});
		canvas.size(1280, 768);
		canvas.save(outDir/(dataset->outputName(i)+ ".png"));
		bar.mark_as_completed();
	}
	return true;
}

int main(int argc, char** argv)
{
	std::cout<<std::setprecision(5)<<std::fixed<<std::setw(3);
	Log::level = Log::INFO;

	Config config;
	argp_parse(&argp, argc, argv, 0, 0, &config);

	if(config.datasetMode == DATASET_INVALID)
	{
		Log(Log::ERROR)<<"You must specify what dataset to use: -d " DATASET_LIST;
		return -1;
	}

	if(config.fileName.empty())
	{
		Log(Log::ERROR)<<"You must specify a valid dataset file to use";
		return 2;
	}

	std::string report;
	switch(config.datasetMode)
	{
		case DATASET_DIR:
		{
			EisDirDataset dataset(config.fileName);
			report = generateReport<EisDirDataset>(&dataset);
			if(!config.imageOutput.empty())
				saveImages<EisDirDataset>(&dataset, config.imageOutput);
			break;
		}
		case DATASET_TAR:
		{
			EisTarDataset dataset(config.fileName);
			report = generateReport<EisTarDataset>(&dataset);
			if(!config.imageOutput.empty())
				saveImages<EisTarDataset>(&dataset, config.imageOutput);
			break;
		}
		case DATASET_TAR_REGRESSION:
		{
			RegressionLoaderTar dataset(config.fileName);
			report = generateReport<RegressionLoaderTar>(&dataset);
			if(!config.imageOutput.empty())
				saveImages<RegressionLoaderTar>(&dataset, config.imageOutput);
			break;
		}
		case DATASET_DIR_REGRESSION:
		{
			RegressionLoaderDir dataset(config.fileName);
			report = generateReport<RegressionLoaderDir>(&dataset);
			if(!config.imageOutput.empty())
				saveImages<RegressionLoaderDir>(&dataset, config.imageOutput);
			break;
		}
		default:
			Log(Log::ERROR)<<"Dataset not implmented";
			break;
	}

	Log(Log::INFO)<<report;
	if(!config.imageOutput.empty())
		Log(Log::INFO)<<"Images saved to "<<config.imageOutput;

	if(!config.reportFileName.empty())
	{
		std::ofstream file(config.reportFileName, std::ios_base::out);
		if(!file.is_open())
		{
			Log(Log::ERROR)<<"Could not open "<<config.reportFileName<<" for writeing\n";
			return 1;
		}
		file<<report;
		file.close();
	}

	return 0;
}

