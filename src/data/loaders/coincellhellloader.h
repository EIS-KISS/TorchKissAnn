#pragma once

#include <eisgenerator/eistype.h>
#include <eisgenerator/model.h>
#include <string>
#include <filesystem>

#include "data/regressiondataset.h"
#include "microtar.h"

class CoinCellHellLoader :
public RegressionDataset<CoinCellHellLoader>
{
	struct Metadata
	{
		long step;
		int substep;
		int cellid;
		int cell_group;
		float temparature;
		float ocv;
		int charge_cycles;
		int thermal_cycles;
		float last_avg_cap;
		long last_avg_step;
		float last_cap;
		long last_cap_step;
		float soc;
		float soc_estimate;
		float cap_estimate;
	};

private:
	virtual torch::data::Example<torch::Tensor, torch::Tensor> getImpl(size_t index) override;

	std::vector<mtar_header_t> files;
	std::filesystem::path path;
	mtar_t tar = {};
	std::vector<int64_t> dataSizes = {};

	eis::EisSpectra loadSpectra(size_t index);
	Metadata loadMetadata(const eis::EisSpectra& spectra);

public:
	explicit CoinCellHellLoader(const std::filesystem::path& pathI);
	CoinCellHellLoader(const CoinCellHellLoader& in);
	CoinCellHellLoader& operator=(const CoinCellHellLoader& in);
	~CoinCellHellLoader();

	virtual size_t outputSize() const override;
	virtual size_t classForIndex(size_t index) override;
	virtual std::string modelStringForClass(size_t classNum) override;
	virtual c10::optional<size_t> size() const override;
	virtual bool isMulticlass() override;
	virtual std::string dataLabel() const override;
	virtual c10::optional<torch::Tensor> freqRange() override;
	virtual std::pair<torch::Tensor, torch::Tensor> getTargetScalesAndBias() override;
	virtual std::vector<std::pair<std::string, int64_t>> extraInputs() override;
};
