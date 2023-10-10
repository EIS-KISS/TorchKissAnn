#include <cstddef>
#include <cstdint>
#include <eisgenerator/eistype.h>
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
#include <vector>
#include <limits>

#include "log.h"
#include "data/generators/eisgendata.h"
#include "data/generators/trashgenerator.h"
#include "data/generators/classextractordataset.h"
#include "data/loaders/dirloader.h"
#include "data/loaders/deduploader.h"
#include "data/eistotorch.h"
#include "options.h"
#include "tensoroperators.h"

template <typename DataSetType>
std::string generateReport(EisDataset<DataSetType>* dataset)
{
	std::stringstream ss;

	ss<<"Name, Size, Classes Count, Mulitclass\n";
	ss<<typeid(DataSetType).name()<<", "<<dataset->size().value()
	  <<", "<<dataset->classesCount()<<", "<<dataset->isMulticlass()<<'\n';

	for(size_t i = 0; i < dataset->classesCount(); ++i)
	{
		ss<<i;
		if(i+1 < dataset->classesCount())
			ss<<", ";
	}
	ss<<'\n';
	for(size_t i = 0; i < dataset->classesCount(); ++i)
	{
		ss<<dataset->modelStringForClass(i);
		if(i+1 < dataset->classesCount())
			ss<<", ";
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
	size_t classesCount = dataset->classesCount();

	for(size_t i = 0; i < classesCount; ++i)
	{
		ClassExtractorDataset<DataSetType> extractedClass(dataset, i);

		sciplot::Plot2D plot;
		plot.xlabel("Re");
		plot.ylabel("Im");

		double min = std::numeric_limits<double>::max();
		double max = std::numeric_limits<double>::min();

		for(size_t j = 0; j < extractedClass.size().value(); ++j)
		{
			torch::data::Example<torch::Tensor, torch::Tensor> example = extractedClass.get(j);
			std::pair<std::valarray<double>, std::valarray<double>> data = torchToValarray<double>(example.data.to(torch::kFloat64));


			double lmin = torch::min(example.data).item().toDouble();
			lmin = lmin - std::abs(min*0.05);
			double lmax = torch::max(example.data).item().toDouble();
			lmax = lmax + std::abs(max*0.05);
			if(lmin < min)
				min = lmin;
			else if(lmax > max)
				max = lmax;
			plot.xrange(min, max);
			plot.yrange(min, max);

			plot.drawCurve(data.first, data.second);
		}
		sciplot::Figure fig({{plot}});
		sciplot::Canvas canvas({{fig}});
		canvas.save(outDir/(dataset->modelStringForClass(i)+ ".png"));
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

	std::string report;
	switch(config.datasetMode)
	{
		case DATASET_GEN:
		{
			EisGeneratorDataset *dataset;
			if(config.fileName.empty())
				dataset = EisGeneratorDataset::newDefaultDataset(100, config.datasetSize, 0, false);
			else
				dataset = new EisGeneratorDataset(config.fileName, config.datasetSize, 100, 0, false, false);
			report = generateReport<EisGeneratorDataset>(dataset);
			if(!config.imageOutput.empty())
				saveImages<EisGeneratorDataset>(dataset, config.imageOutput);
			delete dataset;
			break;
		}
		case DATASET_FDUP:
		{
			if(config.fileName.empty())
			{
				Log(Log::ERROR)<<"You must specify a valid dataset file to use";
				return 2;
			}
			FastDedupDataset dataset(config.fileName);
			report = generateReport<FastDedupDataset>(&dataset);
			if(!config.imageOutput.empty())
				saveImages<FastDedupDataset>(&dataset, config.imageOutput);
			break;
		}
		case DATASET_DIR:
		{
			if(config.fileName.empty())
			{
				Log(Log::ERROR)<<"You must specify a valid dataset file to use";
				return 2;
			}
			EisDirDataset dataset(config.fileName);
			report = generateReport<EisDirDataset>(&dataset);
			if(!config.imageOutput.empty())
				saveImages<EisDirDataset>(&dataset, config.imageOutput);
			break;
		}
		case DATASET_TRASH:
		{
			TrashDataset dataset;
			report = generateReport<TrashDataset>(&dataset);
			if(!config.imageOutput.empty())
				saveImages<TrashDataset>(&dataset, config.imageOutput);
			break;
		}
		default:
			Log(Log::ERROR)<<"You must specify a valid dataset to use: " DATASET_LIST;
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

