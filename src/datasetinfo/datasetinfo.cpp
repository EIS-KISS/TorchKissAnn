#include <ATen/TensorIndexing.h>
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

#include "data/generators/eisgendatanoise.h"
#include "data/passfaildataset.h"
#include "log.h"
#include "data/generators/eisgendata.h"
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

	ss<<"Name,\tSize,\tClass Count,\tMulitclass\n";
	ss<<typeid(DataSetType).name()<<",\t"<<dataset->size().value()
	  <<",\t"<<dataset->classesCount()<<",\t"<<dataset->isMulticlass()<<'\n';

	for(size_t i = 0; i < dataset->classesCount(); ++i)
	{
		ss<<i;
		if(i+1 < dataset->classesCount())
			ss<<",\t";
	}
	ss<<'\n';
	for(size_t i = 0; i < dataset->classesCount(); ++i)
	{
		ss<<dataset->modelStringForClass(i);
		if(i+1 < dataset->classesCount())
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
	size_t classesCount = dataset->classesCount();

	for(size_t i = 0; i < classesCount; ++i)
	{
		Log(Log::INFO)<<"Processing class "<<i+1<<" of "<<classesCount;
		ClassExtractorDataset<DataSetType> extractedClass(dataset, i);

		sciplot::Plot2D plot;
		plot.xlabel("Re");
		plot.ylabel("Im");

		double xmin = std::numeric_limits<double>::max();
		double xmax = std::numeric_limits<double>::min();
		double ymin = std::numeric_limits<double>::max();
		double ymax = std::numeric_limits<double>::min();

		for(size_t j = 0; j < extractedClass.size().value(); ++j)
		{
			Log(Log::INFO)<<"Processing spectra "<<j+1<<" of "<<extractedClass.size().value();
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
			plot.yrange(ymin, ymax);
			plot.legend().hide();

			plot.drawCurve(data.first, data.second);
		}
		sciplot::Figure fig({{plot}});
		sciplot::Canvas canvas({{fig}});
		canvas.size(1280, 768);
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
				dataset = EisGeneratorDataset::newDefaultDataset(100, config.datasetSize, 1, false);
			else
				dataset = new EisGeneratorDataset(config.fileName, config.datasetSize, 100, 1, false, false);
			report = generateReport<EisGeneratorDataset>(dataset);
			if(!config.imageOutput.empty())
				saveImages<EisGeneratorDataset>(dataset, config.imageOutput);
			delete dataset;
			break;
		}
		case DATASET_GEN_NOISE:
		{
			EisGeneratorDatasetNoise *dataset;
			if(config.fileName.empty())
				dataset = EisGeneratorDatasetNoise::newDefaultDataset(100, config.datasetSize, false);
			else
				dataset = new EisGeneratorDatasetNoise(config.fileName, config.datasetSize, 100, false);
			report = generateReport<EisGeneratorDatasetNoise>(dataset);
			if(!config.imageOutput.empty())
				saveImages<EisGeneratorDatasetNoise>(dataset, config.imageOutput);
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
		case DATASET_PASSFAIL:
		{
			EisGeneratorDataset *origDataset = EisGeneratorDataset::newDefaultDataset(100, config.datasetSize, 0, false);
			PassFaillDataset<EisGeneratorDataset> dataset(origDataset);
			report = generateReport<PassFaillDataset<EisGeneratorDataset>>(&dataset);
			if(!config.imageOutput.empty())
				saveImages<PassFaillDataset<EisGeneratorDataset>>(&dataset, config.imageOutput);
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

