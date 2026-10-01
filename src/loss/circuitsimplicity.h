/* * TorchKissAnn - A collection of tools to train various types of Machine learning
 * algorithms on various types of EIS data
 * Copyright (C) 2025 Carl Klemm <carl@uvos.xyz>
 *
 * This file is part of TorchKissAnn.
 *
 * TorchKissAnn is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * TorchKissAnn is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with TorchKissAnn.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "torchph.h"

/**
 * The label degeneracy of an equivalent circuit.
 *
 * A circuit whose elements all matter describes its spectra uniquely, but a circuit with an element
 * that contributes nothing is described equally well by the circuit with that element removed:
 * "r-rc-rc" whose last block sits at a few milliohms is "r-rc", and a classifier asked to name the
 * generating circuit of such a spectra is being asked for something the spectra does not contain.
 * That is the classification counterpart of the parameter degeneracy PermutationMseLoss handles for
 * regression, and like it, it is a property of the sample rather than of the network.
 *
 * Where the parameter degeneracy is resolved by fitting, this one is not: asking whether some other
 * circuit explains a spectra is a global optimisation, and a Levenberg-Marquardt run per training
 * sample is far too expensive to sit inside a loss. It does not have to be. The training sample
 * knows the circuit that generated it, including the parameter values, which the EISF header
 * carries, so the question is not "can anything explain this spectra" but "does this circuit still
 * describe its own spectra without one of its elements". That is a contribution test, and
 * eis::ParallelSerial::contributionRatio() answers it exactly: for a series group it is the removed
 * element's impedance over the total, for a parallel group the total over the removed branch, in
 * both cases the relative change of |Z| the removal causes. A handful of complex evaluations per
 * sample, no optimiser, no gradient.
 */
namespace circuit
{

struct SimplicityOptions
{
	/**
	 * Relative change of |Z| below which a reduced circuit still describes the generating circuit's
	 * spectra over the measured band: max over omega of |Z_reduced(omega) - Z_full(omega)| /
	 * |Z_full(omega)|. A circuit whose worst deviation stays under this is one the spectra can not
	 * rule out, so the network should not be blamed for naming it.
	 *
	 * The quantity is a ratio of two impedances of the same circuit and therefore bounded by the
	 * circuit itself, not by the threshold: an element that dominates the spectra somewhere drives it
	 * towards 1. Values above 1 accept every reduction the topology allows, which is not a useful
	 * score, so keep this well inside (0,1).
	 */
	double contributionThreshold = 0.01;
	/// How many elements may be missing at once. Two covers a circuit that has grown both a stray
	/// series resistance and a stray time constant; more is combinatorics for its own sake.
	size_t maxDeletions = 2;
	/// Ceiling on the number of removal combinations examined per sample, so a circuit with many
	/// elements cannot turn the label pass into a search.
	size_t maxCandidates = 64;
};

/**
 * Whether every element of a model string carries an explicit {parameter} block.
 *
 * The contribution test below compares impedances, so it needs the values, not just the topology.
 * eisgenerator silently substitutes a sweep range for a bare element and resolveSteps(0) then pins
 * it to the start of that range, which turns a header spelling like "rp-p" into an arbitrary circuit
 * whose element contributions mean nothing. Simulated datasets write the values they sampled;
 * datasets assembled from measurements usually carry the topology alone.
 *
 * Returns true for the empty string, which is not a circuit at all and is the caller's "no circuit"
 * case, and for anything that is not a parseable circuit, which the caller rejects on its own.
 */
bool modelCarriesParameterValues(const std::string& modelStr);

/**
 * Topology of a circuit, ignoring the parameter values and the way the string happens to be
 * bracketed. "r-rc-rc", "(r-(rc)-(rc))" and "r-r-c-r-c" all reduce to the same key, so a circuit
 * this analysis produces can be looked up in a dataset's class table by what it is rather than by
 * how its label is spelled. Nested groups of the same kind are spliced into their parent, which is
 * electrically exact, and a group left with a single element is that element.
 *
 * Returns an empty string for anything eis::Model can not parse, which includes the placeholder
 * labels a dataset carries for spectra with no circuit in their header.
 */
std::string canonicalKey(const std::string& modelStr);

/**
 * The canonical keys of the circuits obtainable from a generating circuit by removing up to
 * options.maxDeletions elements without changing its impedance by more than
 * options.contributionThreshold at any frequency in omegas, fewest elements first.
 *
 * The generating circuit must carry its parameter values, as the model string in a simulated
 * dataset's EISF header does; a bare "r-rc-rc" says nothing about which element is negligible and
 * modelCarriesParameterValues() rejects it. Circuits that can not be parsed, that leave nothing
 * behind and that are not actually simpler than the original are left out. So is the original
 * itself.
 */
std::vector<std::string> reducibleCircuitKeys(const std::string& generatingModel, const std::vector<double>& omegas,
		const SimplicityOptions& options = SimplicityOptions());

/**
 * The frequency axis out of the tensor EisSpectraDataset::frequencies() hands back. That tensor
 * holds one entry per network input, so with a canonical grid it repeats the grid once for the real
 * and once for the imaginary part; the contribution test wants the grid itself, once.
 */
std::vector<double> uniqueOmegas(const torch::Tensor& frequencies);

/**
 * The class table of a classification dataset, indexed by topology.
 *
 * Built once from the dataset's output labels. Look a circuit up with find() to get the class the
 * network would have to predict for it, or ask for labelWeights() to get the whole label of one
 * sample: the generating class plus every class that describes the same spectra.
 */
class LabelSpace
{
	std::vector<std::string> keys;
	std::vector<size_t> elementCounts;
	std::map<std::string, int64_t> indexByKey;

	public:
		void build(const std::vector<std::string>& labels);

		bool empty() const
		{
			return keys.empty();
		}

		size_t size() const
		{
			return keys.size();
		}

		/// The class holding this topology, or -1 when the space has no such class.
		int64_t find(const std::string& key) const;

		/// How many circuit elements a class has, used to decide which of two equivalent classes is
		/// the simpler one. 0 for a class that is not a circuit at all.
		size_t elementCount(int64_t classId) const;

		/**
		 * The label of one sample as a weight per class: 1 on the generating class and on every
		 * class that describes the same spectra, and simplicityPreference more on whichever of
		 * those has the fewest elements. Sums to more than one; the loss normalises.
		 *
		 * Returns an empty vector when the generating circuit can not be parsed, which leaves the
		 * caller to fall back to the plain class label.
		 */
		std::vector<float> labelWeights(const std::string& generatingModel, int64_t generatingClass,
				const std::vector<double>& omegas, double simplicityPreference,
				const SimplicityOptions& options = SimplicityOptions()) const;
};

/**
 * Run the analysis over a whole dataset in one pass.
 *
 * @param labels the class table, class id to label, as a loader's outputName() gives it
 * @param generatingModels one model string per sample, with the parameter values from its header
 * @param classIndexes the class of each sample, same order as generatingModels
 * @return one weight vector of labels.size() entries per sample, or empty when the class table is
 *         empty. A sample whose circuit can not be analysed comes back as a one hot label.
 */
std::vector<std::vector<float>> computeLabelWeights(const std::vector<std::string>& labels,
		const std::vector<std::string>& generatingModels, const std::vector<size_t>& classIndexes,
		const std::vector<double>& omegas, double simplicityPreference,
		const SimplicityOptions& options = SimplicityOptions());

}
