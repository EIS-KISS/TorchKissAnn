#include "drt.h"
#include "T2_inversion.h"

std::vector<fvalue> calcDrt(const std::vector<eis::DataPoint>& data)
{
	double* cR = new double[data.size()];
	double* cIm = new double[data.size()];
	double* freq = new double[data.size()];

	for(size_t i = 0; i < data.size(); ++i)
	{
		cR[i] = data[i].im.real();
		cIm[i] = data[i].im.imag();
		freq[i] = data[i].omega;
	}

	constexpr int nDrt = 20;
	double T2[nDrt];
	logspace(T2, -4, 0.7, nDrt);
	double T2dist[nDrt + 1];
	double syn_data[data.size()];

	double stdev;
	double snr = calculateStdDevAndSnr(cR, cIm, data.size(), stdev);

	T2inversion(T2, cR, cIm, freq, 1, T2dist, syn_data, data.size(), stdev, nDrt);

	std::vector<fvalue> drt;
	drt.reserve(nDrt);

	for(size_t i = 0; i < nDrt; ++i)
		drt.push_back(T2dist[i]);
	return drt;
}
