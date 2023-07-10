#pragma once

#include <memory>
#include <cmath>
#include <string>

#include "Network.h"
#include "config.h"

#include <fstream>
#include <chrono> // time since 1970

inline int binarySearch(float* proba, float value, int size) {
	int inf = 0;
	int sup = size - 1;

	if (proba[inf] > value) {
		return inf;
	}

	int mid;
	int max_iter = 15;
	while (sup - inf >= 1 && max_iter--) {
		mid = (sup + inf) / 2;
		if (proba[mid] < value && value <= proba[mid + 1]) {
			return mid + 1;
		}
		else if (proba[mid] < value) {
			inf = mid;
		}
		else {
			sup = mid;
		}
	}
	return 0; // not necessarily a failure, since floating point approximation prevents the sum from reaching 1.
	//throw "Binary search failure !";
}

// Contains the parameters for population evolution. They can be changed at each step.
struct PopulationEvolutionParameters {

	// Both values should be < 1.0 , safe value is 0.0 .  
	// ".first " influences the probability of each specimen to be present once in the next generation. 
	// ".second" influences the probability of each specimen to take a spot left empty by a specimen that did not make it
	std::pair<float, float> selectionPressure;


	// Minimum at 1. If = 1, mutated clone of the parent. No explicit maximum, but bounded by nSpecimens, 
	// and MAX_MATING_DEPTH implicitly. Cost O( n log(n) ).
	int nParents;

	//defaults:
	PopulationEvolutionParameters() {
		selectionPressure = { -10.0f, 0.0f };
		nParents = 10;
	}
};


struct PhylogeneticNode
{
	PhylogeneticNode* parent;
	int networkIndice;
	std::vector<PhylogeneticNode*> children;

	// TODO be careful, it wont work as intended if there are multiple Population . 
	// Should not happen. Set in Population::setEvolutionParameters.
	static int maxListSize;

	PhylogeneticNode() {};

	PhylogeneticNode(PhylogeneticNode* parent, int networkIndice) :
		networkIndice(networkIndice), parent(parent) {};

	bool addToList(std::vector<int>& list, int depth) {
		if (depth == 0) {
			if (list.size() == maxListSize) {
				return false;
			}
			list.push_back(networkIndice);
			return true;
		}
		else {
			for (int i = 0; i < children.size(); i++) {
				if (!children[i]->addToList(list, depth - 1)) {
					return false;
				}
			}
			return true;
		}
	}

	void erase(int depth) {
		if (depth == MAX_MATING_DEPTH) return;

		if (parent->children.size() == 1) {
			parent->children.resize(0);
			parent->erase(depth + 1);
		}
		else {
			int s = (int) parent->children.size() - 1;
			for (int i = 0; i < s; i++) {
				if (parent->children[i] == this) {
					parent->children[i] = parent->children[s];
				}
			}
			parent->children.pop_back();
		}
	}
};



// A group of a fixed number of individuals, optimized with a genetic algorithm.
class Population {

public:	
	// testing features
	void test();

	~Population();

	void step();

	Population(int IN_SIZE, int OUT_SIZE, int nSpecimens);

	void createOffsprings();

	void setEvolutionParameters(PopulationEvolutionParameters params) {
		this->selectionPressure = params.selectionPressure;
		this->nParents = params.nParents;

		PhylogeneticNode::maxListSize = params.nParents;
	}

	
	void saveFittestSpecimen() const 
	{
		uint64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::system_clock::now().time_since_epoch()).count();

		std::ofstream os("models\\topNet_" + std::to_string(ms) + "_" + std::to_string(evolutionStep) + ".renon", std::ios::binary);
		networks[fittestSpecimen]->save(os);
	}

	// Indice in the networks list of the fittest specimen at this step.
	int fittestSpecimen;

	// Current number of generations since initialization.
	int evolutionStep;

private:
	float* datasetX, *datasetY;
	float evaluateNetOnDataset(Network* n);

	//  = PhylogeneticNode[MAX_N_PARENTS][nSpecimens]
	PhylogeneticNode* phylogeneticTree;

	// Finds the secondary parents, computes the coefficients, and creates the interpolated child.
	Network* createChild(PhylogeneticNode* primaryParent);

	
	// Current size of the networks and fitness arrays. Must be a multiple of N_THREADS.
	int nSpecimens;
		
	// Size nSpecimens
	std::vector<Network*> networks;
	
	// The vector of fitness per specimen.
	std::vector<float> fitnesses;

	// The scores of the specimens at this step, as output by the trials.
	std::vector<float> rawScores;

	// Raw scores after a transformation that depends on the whole population's performance. 
	// Like ranking, or normalization.
	std::vector<float> batchTransformedScores;


	// EVOLUTION PARAMETERS: 
	
	// Set with a PopulationEvolutionParameters struct. Description in the struct definition.
	std::pair<float, float> selectionPressure;

	// Set with a PopulationEvolutionParameters struct. Description in the struct definition.
	int nParents;
};