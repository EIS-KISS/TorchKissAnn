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

#include "circuitsimplicity.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <eisgenerator/componant/componant.h>
#include <eisgenerator/componant/paralellseriel.h>
#include <eisgenerator/log.h>
#include <eisgenerator/model.h>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <utility>

#include "log.h"

namespace
{

struct Node
{
	char kind = 0;   // 'S' series group, 'P' parallel group, 0 leaf
	std::string leaf;
	std::vector<Node> children;
	uint32_t id = 0;
	bool deletable = false;
	/// The component this node mirrors, only set on leaves.
	eis::Componant* componant = nullptr;
	/// The impedance of this node at every frequency of the analysis grid, filled in by
	/// cacheImpedance() on the full tree. A reduction reuses it for every subtree it leaves intact.
	std::vector<std::complex<fvalue>> z;
};

std::string stripParameters(const std::string& in)
{
	std::string out;
	int depth = 0;

	for(char c : in)
	{
		if(c == '{' || c == '[' || c == '<')
			++depth;
		if(depth == 0)
			out.push_back(c);
		if(c == '}' || c == ']' || c == '>')
			--depth;
	}
	return out;
}

std::vector<eis::Componant*>* childrenOf(eis::Componant* componant)
{
	if(eis::Serial* serial = dynamic_cast<eis::Serial*>(componant))
		return &serial->componants;
	if(eis::Parallel* parallel = dynamic_cast<eis::Parallel*>(componant))
		return &parallel->componants;
	return nullptr;
}

std::string keyOf(const struct Node& node);

/**
 * The key parts of node as seen from inside a group of groupKind. A group of the same kind as its
 * parent is spliced in, because (a-b)-c is the same circuit as a-b-c and the two spellings must not
 * produce two different classes.
 */
void collectKeyParts(const Node& node, char groupKind, std::vector<std::string>& parts)
{
	if(node.kind == groupKind)
	{
		for(const Node& child : node.children)
			collectKeyParts(child, groupKind, parts);
		return;
	}

	std::string key = keyOf(node);
	if(!key.empty())
		parts.push_back(std::move(key));
}

std::string keyOf(const Node& node)
{
	if(node.kind == 0)
		return stripParameters(node.leaf);

	std::vector<std::string> parts;
	for(const Node& child : node.children)
		collectKeyParts(child, node.kind, parts);

	if(parts.empty())
		return std::string();
	if(parts.size() == 1)
		return parts.front();

	std::string out(1, node.kind);
	out.push_back('[');
	for(size_t i = 0; i < parts.size(); ++i)
	{
		if(i != 0)
			out.push_back(',');
		out.append(parts[i]);
	}
	out.push_back(']');
	return out;
}

size_t leafCount(const Node& node)
{
	if(node.kind == 0)
		return 1;
	size_t count = 0;
	for(const Node& child : node.children)
		count += leafCount(child);
	return count;
}

Node reduce(const Node& in, const std::set<uint32_t>& removed)
{
	Node out;
	out.kind = in.kind;
	out.leaf = in.leaf;
	out.id = in.id;
	out.componant = in.componant;

	for(const Node& child : in.children)
	{
		if(removed.count(child.id) != 0)
			continue;
		out.children.push_back(reduce(child, removed));
	}
	return out;
}

Node buildNode(eis::Componant* componant, uint32_t& nextId)
{
	Node node;
	node.id = nextId++;

	std::vector<eis::Componant*>* kids = childrenOf(componant);
	if(kids == nullptr || dynamic_cast<eis::ParallelSerial*>(componant) == nullptr)
	{
		node.leaf = componant->getComponantString(true);
		node.componant = componant;
		return node;
	}

	node.kind = dynamic_cast<eis::Serial*>(componant) != nullptr ? 'S' : 'P';

	for(eis::Componant* kid : *kids)
	{
		Node child = buildNode(kid, nextId);
		// a group of one is that element, so only a group with siblings can lose a child and stay a
		// circuit; whether losing it is allowed is the impedance test below, not a per element gate
		child.deletable = kids->size() > 1;
		node.children.push_back(std::move(child));
	}
	return node;
}

std::complex<fvalue> reciprocal(std::complex<fvalue> z)
{
	fvalue magnitude2 = std::norm(z);
	if(magnitude2 == 0)
		return std::complex<fvalue>(std::numeric_limits<fvalue>::infinity(), 0);
	return std::complex<fvalue>(z.real()/magnitude2, -z.imag()/magnitude2);
}

/**
 * The impedance of node at every grid frequency, bottom up.
 *
 * Every node is evaluated exactly once, so this costs the same as a single sweep of the whole
 * circuit and gives the reductions below their untouched subtrees for free.
 */
void cacheImpedance(Node& node, const std::vector<double>& omegas)
{
	node.z.assign(omegas.size(), std::complex<fvalue>(0, 0));

	if(node.kind == 0)
	{
		for(size_t i = 0; i < omegas.size(); ++i)
			node.z[i] = node.componant->execute(omegas[i]);
		return;
	}

	for(Node& child : node.children)
		cacheImpedance(child, omegas);

	for(size_t i = 0; i < omegas.size(); ++i)
	{
		std::complex<fvalue> acc(0, 0);
		for(const Node& child : node.children)
			acc += (node.kind == 'S') ? child.z[i] : reciprocal(child.z[i]);
		node.z[i] = (node.kind == 'S') ? acc : reciprocal(acc);
	}
}

/// True when node or anything below it is in removed; every group on the way is marked dirty.
bool markDirty(const Node& node, const std::set<uint32_t>& removed, std::set<uint32_t>& dirty)
{
	if(node.kind == 0)
		return removed.count(node.id) != 0;

	bool affected = false;
	for(const Node& child : node.children)
		if(markDirty(child, removed, dirty))
			affected = true;

	if(affected)
		dirty.insert(node.id);
	return affected || removed.count(node.id) != 0;
}

/// The impedance of the tree with removed taken out at grid index i, or false when node itself is
/// gone. Groups the removal did not touch come straight out of the cache.
bool reducedImpedance(const Node& node, const std::set<uint32_t>& removed, const std::set<uint32_t>& dirty,
		size_t i, std::complex<fvalue>& out)
{
	if(removed.count(node.id) != 0)
		return false;
	if(node.kind == 0 || dirty.count(node.id) == 0)
	{
		out = node.z[i];
		return true;
	}

	std::complex<fvalue> acc(0, 0);
	bool any = false;
	for(const Node& child : node.children)
	{
		std::complex<fvalue> kid(0, 0);
		if(not reducedImpedance(child, removed, dirty, i, kid))
			continue;
		any = true;
		acc += (node.kind == 'S') ? kid : reciprocal(kid);
	}

	out = (node.kind == 'S') ? acc : reciprocal(acc);
	return any;
}

/**
 * The worst relative change of |Z| the removal causes anywhere on the grid.
 *
 * This is the whole criterion: a reduction whose impedance stays within the threshold of the
 * generating circuit is one the spectra can not tell apart from it, however many elements it dropped.
 */
fvalue relativeDeviation(const Node& root, const std::set<uint32_t>& removed, const std::set<uint32_t>& dirty,
		const std::vector<double>& omegas)
{
	fvalue worst = 0;
	for(size_t i = 0; i < omegas.size(); ++i)
	{
		std::complex<fvalue> reduced(0, 0);
		if(not reducedImpedance(root, removed, dirty, i, reduced))
			return std::numeric_limits<fvalue>::infinity();

		fvalue full = std::abs(root.z[i]);
		if(not std::isfinite(full) || full == 0)
			continue;

		worst = std::max(worst, std::abs(reduced - root.z[i])/full);
	}
	return worst;
}

void collectDeletable(const Node& node, std::vector<uint32_t>& ids)
{
	if(node.deletable)
		ids.push_back(node.id);
	for(const Node& child : node.children)
		collectDeletable(child, ids);
}

bool nextCombination(std::vector<size_t>& c, size_t n)
{
	const size_t k = c.size();
	for(size_t i = k; i-- > 0; )
	{
		if(c[i] != n - k + i)
		{
			++c[i];
			for(size_t j = i+1; j < k; ++j)
				c[j] = c[j-1]+1;
			return true;
		}
	}
	return false;
}

class EisLogMute
{
	eis::Log::Level previous;
	public:
		EisLogMute(): previous(eis::Log::level)
		{
			if(eis::Log::level < eis::Log::ERROR)
				eis::Log::level = eis::Log::ERROR;
		}
		~EisLogMute()
		{
			eis::Log::level = previous;
		}
};

}

std::string circuit::canonicalKey(const std::string& modelStr)
{
	if(modelStr.empty())
		return std::string();

	EisLogMute mute;

	try
	{
		eis::Model model(modelStr);
		if(!model.isReady())
			return std::string();
		uint32_t nextId = 0;
		return keyOf(buildNode(model.getModelRoot(), nextId));
	}
	catch(const std::exception& ex)
	{
		Log(Log::DEBUG)<<"Not a circuit: "<<modelStr<<": "<<ex.what();
		return std::string();
	}
}

bool circuit::modelCarriesParameterValues(const std::string& modelStr)
{
	int depth = 0;

	for(size_t i = 0; i < modelStr.size(); ++i)
	{
		char c = modelStr[i];
		if(c == '{' || c == '[' || c == '<')
		{
			++depth;
			continue;
		}
		if(c == '}' || c == ']' || c == '>')
		{
			--depth;
			continue;
		}
		if(depth > 0 || !eis::Componant::isValidComponantChar(c))
			continue;

		size_t next = i+1;
		while(next < modelStr.size() && std::isspace(static_cast<unsigned char>(modelStr[next])))
			++next;
		if(next >= modelStr.size() || modelStr[next] != '{')
			return false;
	}
	return true;
}

std::vector<std::string> circuit::reducibleCircuitKeys(const std::string& generatingModel, const std::vector<double>& omegas,
		const SimplicityOptions& options)
{
	std::vector<std::string> out;
	if(generatingModel.empty() || omegas.empty())
		return out;
	// without the values there is nothing to compare impedances against, and the sweep range
	// eisgenerator substitutes would rate the circuit by numbers nobody measured
	if(!circuit::modelCarriesParameterValues(generatingModel))
		return out;

	EisLogMute mute;

	try
	{
		eis::Model model(generatingModel);
		if(!model.isReady())
			return out;
		model.resolveSteps(0);

		uint32_t nextId = 0;
		Node root = buildNode(model.getModelRoot(), nextId);
		cacheImpedance(root, omegas);

		const std::string baseKey = keyOf(root);
		const size_t baseElements = leafCount(root);
		if(baseKey.empty())
			return out;

		std::vector<uint32_t> deletable;
		collectDeletable(root, deletable);
		if(deletable.size() > 16)
			deletable.resize(16);

		std::set<std::string> seen;
		std::vector<std::pair<std::string, size_t>> found;
		size_t candidates = 0;

		for(size_t k = 1; k <= options.maxDeletions && k <= deletable.size(); ++k)
		{
			std::vector<size_t> combination(k);
			std::iota(combination.begin(), combination.end(), 0);
			do
			{
				if(candidates++ >= options.maxCandidates)
					break;

				std::set<uint32_t> removed;
				for(size_t pick : combination)
					removed.insert(deletable[pick]);

				// the key is cheap and the impedance sweep is not, so a topology that is no simpler
				// than the original never reaches the deviation test
				Node reduced = reduce(root, removed);
				const std::string key = keyOf(reduced);
				const size_t elements = leafCount(reduced);
				if(key.empty() || key == baseKey || elements >= baseElements || seen.count(key) != 0)
					continue;

				std::set<uint32_t> dirty;
				markDirty(root, removed, dirty);
				if(relativeDeviation(root, removed, dirty, omegas) >= options.contributionThreshold)
					continue;

				seen.insert(key);
				found.emplace_back(key, elements);
			} while(nextCombination(combination, deletable.size()));
		}

		std::stable_sort(found.begin(), found.end(), [](const std::pair<std::string, size_t>& a, const std::pair<std::string, size_t>& b)
		{
			return a.second < b.second;
		});
		out.reserve(found.size());
		for(const std::pair<std::string, size_t>& entry : found)
			out.push_back(entry.first);
	}
	catch(const std::exception& ex)
	{
		Log(Log::DEBUG)<<"Simplicity analysis failed for "<<generatingModel<<": "<<ex.what();
		return std::vector<std::string>();
	}

	return out;
}

std::vector<double> circuit::uniqueOmegas(const torch::Tensor& frequencies)
{
	if(!frequencies.defined() || frequencies.numel() < 2)
		return std::vector<double>();

	torch::Tensor omega = frequencies.detach().to(torch::kFloat64).cpu().reshape({frequencies.numel()});

	const int64_t half = omega.numel()/2;
	if(omega.numel() % 2 == 0 && torch::equal(omega.narrow(0, 0, half), omega.narrow(0, half, half)))
		omega = omega.narrow(0, 0, half);

	std::vector<double> out(omega.numel());
	std::memcpy(out.data(), omega.data_ptr<double>(), out.size()*sizeof(double));
	return out;
}

void circuit::LabelSpace::build(const std::vector<std::string>& labels)
{
	keys.clear();
	elementCounts.clear();
	indexByKey.clear();

	keys.reserve(labels.size());
	for(const std::string& label : labels)
	{
		const std::string key = circuit::canonicalKey(label);
		const int64_t index = static_cast<int64_t>(keys.size());
		size_t elements = 0;
		if(!key.empty())
		{
			elements = std::count(key.begin(), key.end(), 'r') + std::count(key.begin(), key.end(), 'c')
				+ std::count(key.begin(), key.end(), 'p') + std::count(key.begin(), key.end(), 'l')
				+ std::count(key.begin(), key.end(), 'w') + std::count(key.begin(), key.end(), 't')
				+ std::count(key.begin(), key.end(), 'o');
			if(indexByKey.count(key) == 0)
				indexByKey.emplace(key, index);
		}
		elementCounts.push_back(elements);
		keys.push_back(key);
	}
}

int64_t circuit::LabelSpace::find(const std::string& key) const
{
	if(key.empty())
		return -1;
	std::map<std::string, int64_t>::const_iterator search = indexByKey.find(key);
	if(search == indexByKey.end())
		return -1;
	return search->second;
}

size_t circuit::LabelSpace::elementCount(int64_t classId) const
{
	if(classId < 0 || static_cast<size_t>(classId) >= elementCounts.size())
		return 0;
	return elementCounts[classId];
}

std::vector<float> circuit::LabelSpace::labelWeights(const std::string& generatingModel, int64_t generatingClass,
		const std::vector<double>& omegas, double simplicityPreference, const SimplicityOptions& options) const
{
	std::vector<float> weights(keys.size(), 0.0f);
	if(weights.empty())
		return weights;

	weights[generatingClass >= 0 && static_cast<size_t>(generatingClass) < weights.size() ?
		static_cast<size_t>(generatingClass) : 0] = 1.0f;

	if(generatingModel.empty())
		return weights;

	int64_t simplest = generatingClass;
	size_t simplestCount = elementCount(generatingClass);
	if(simplestCount == 0)
		return weights;

	for(const std::string& key : reducibleCircuitKeys(generatingModel, omegas, options))
	{
		const int64_t classId = find(key);
		if(classId < 0 || classId == generatingClass)
			continue;
		weights[classId] = 1.0f;
		if(elementCount(classId) < simplestCount)
		{
			simplest = classId;
			simplestCount = elementCount(classId);
		}
	}

	if(simplest != generatingClass && simplicityPreference > 0.0)
		weights[simplest] += static_cast<float>(simplicityPreference);

	return weights;
}

std::vector<std::vector<float>> circuit::computeLabelWeights(const std::vector<std::string>& labels,
		const std::vector<std::string>& generatingModels, const std::vector<size_t>& classIndexes,
		const std::vector<double>& omegas, double simplicityPreference, const SimplicityOptions& options)
{
	std::vector<std::vector<float>> out;
	if(labels.empty() || generatingModels.size() != classIndexes.size())
		return out;

	LabelSpace space;
	space.build(labels);

	if(omegas.empty())
		Log(Log::WARN)<<"No frequency axis for the simplicity analysis, every sample keeps its generating circuit as its only label";

	// The criterion compares impedances, so a header that spells the circuit without its values can
	// not be judged at all: eisgenerator substitutes the start of its default sweep range for every
	// bare element, and every spectra of a class would then be scored against the same made up
	// circuit, giving a number that has nothing to do with the measurement.
	for(const std::string& model : generatingModels)
	{
		if(model.empty() || modelCarriesParameterValues(model) || circuit::canonicalKey(model).empty())
			continue;
		Log(Log::ERROR)<<"the simplicity loss compares the impedance of the generating circuit with that of its reductions, which needs the parameter values, but a spectra carries the bare topology "<<model<<" in its header";
		return out;
	}

	out.reserve(generatingModels.size());
	size_t ambiguous = 0;
	for(size_t i = 0; i < generatingModels.size(); ++i)
	{
		out.push_back(space.labelWeights(generatingModels[i], static_cast<int64_t>(classIndexes[i]),
					omegas, simplicityPreference, options));
		if(out.back().size() > 1 && std::count(out.back().begin(), out.back().end(), 0.0f) != static_cast<int64_t>(out.back().size()-1))
			++ambiguous;
	}

	Log(Log::INFO)<<"Simplicity labels: "<<ambiguous<<" of "<<out.size()<<" spectra are described by more than one of the "
		<<labels.size()<<" classes, accepting any reduction that moves |Z| by less than "<<options.contributionThreshold
		<<" of the generating circuit everywhere on a grid of "<<omegas.size()<<" frequencies, simplest accepted class weighted +"
		<<simplicityPreference;

	return out;
}
