#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <fstream>

#include "log.h"
#include "data/generators/eisgendata.h"
#include "data/generators/trashgenerator.h"
#include "data/loaders/dirloader.h"
#include "data/loaders/deduploader.h"
#include "options.h"
#include "tensoroperators.h"

template <typename DataSetType>
std::string generateReport(EisDataset<DataSetType>* dataset)
{
	std::stringstream ss;

	ss<<"Name,\tSize,\tClasses Count,\tMulitclass\n";
	ss<<typeid(DataSetType).name()<<",\t"<<dataset->size().value()
	  <<",\t"<<dataset->classesCount()<<",\t"<<dataset->isMulticlass()<<'\n';

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
			break;
		}
		case DATASET_TRASH:
		{
			TrashDataset dataset;
			report = generateReport<TrashDataset>(&dataset);
			break;
		}
		default:
			Log(Log::ERROR)<<"You must specify a valid dataset to use: " DATASET_LIST;
			break;
	}

	Log(Log::INFO)<<report;

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

