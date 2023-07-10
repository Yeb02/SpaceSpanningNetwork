#pragma once

#include <iostream>
#include <algorithm> // std::sort

#include "Population.h"
#include "Random.h"


int PhylogeneticNode::maxListSize = 0;

// src is unchanged.
void normalizeArray(float* src, float* dst, int size) {
	float avg = 0.0f;
	for (int i = 0; i < size; i++) {
		avg += src[i];
	}
	avg /= (float)size;
	float variance = 0.0f;
	for (int i = 0; i < size; i++) {
		dst[i] = src[i] - avg;
		variance += dst[i] * dst[i];
	}
	if (variance < .001f) return;
	float InvStddev = 1.0f / sqrtf(variance / (float) size);
	for (int i = 0; i < size; i++) {
		dst[i] *= InvStddev;
	}
}

// src is unchanged. Results in [-1, 1]
void rankArray(float* src, float* dst, int size) {
	std::vector<int> positions(size);
	for (int i = 0; i < size; i++) {
		positions[i] = i;
	}
	// sort position by ascending value.
	std::sort(positions.begin(), positions.end(), [src](int a, int b) -> bool
		{
			return src[a] < src[b];
		}
	);
	float invSize = 1.0f / (float)size;
	for (int i = 0; i < size; i++) {
		// linear in [-1,1], -1 for the worst specimen, 1 for the best
		float positionValue = (float)(2 * i - size) * invSize;
		// arbitrary, to make it a bit more selective. 
		positionValue = 1.953f * powf(positionValue * .8f, 3.0f);

		dst[positions[i]] = positionValue;
	}
	return;
}

Population::Population(int IN_SIZE, int OUT_SIZE, int nSpecimens) :
	nSpecimens(nSpecimens)
{
	rawScores.resize(nSpecimens);
	batchTransformedScores.resize(nSpecimens);
	networks.resize(nSpecimens);
	fitnesses.resize(nSpecimens);
	for (int i = 0; i < nSpecimens; i++) {
		networks[i] = new Network(IN_SIZE, OUT_SIZE);
	}
	
	PopulationEvolutionParameters defaultParams;
	setEvolutionParameters(defaultParams);

	phylogeneticTree = new PhylogeneticNode[MAX_MATING_DEPTH * nSpecimens];
	for (int i = 0; i < nSpecimens; i++) {
		phylogeneticTree[i].children.resize(0);
		phylogeneticTree[i].networkIndice = i;
		phylogeneticTree[i].parent = nullptr;
	}

	fittestSpecimen = 0;
	evolutionStep = 1; // starting at 1 is important for the phylogeneticTree.

	datasetY = new float[networks[0]->outputSize * N_YS];
	for (int k = 0; k < networks[0]->outputSize * N_YS; k++)
	{
		datasetY[k] = std::clamp(NORMAL_01 * .3f, -1.0f, 1.0f);
	}
	datasetX = new float[networks[0]->inputSize * N_YS];
	for (int k = 0; k < networks[0]->inputSize * N_YS; k++)
	{
		datasetX[k] = NORMAL_01;
	}
}

Population::~Population() {
	for (const Network* n : networks) {
		delete n;
	}
	delete[] phylogeneticTree;
	delete[] datasetX;
	delete[] datasetY;
}

float Population::evaluateNetOnDataset(Network* n) {
	
	float score = 0.0f;

	for (int j = 0; j < N_YS; j++) {
		score -= n->forward(&datasetX[j*n->inputSize], &datasetY[j * n->outputSize], NO_GRAD); // - because the higher the distance, the worst the net is.
	}
	return score / (float)N_YS;
}


void Population::test()
{
	int nTests = 20;

	float* X = new float[networks[0]->inputSize];
	float* Y = new float[networks[0]->outputSize];

	float* s = new float[nTests];

	float avg = 0.0f, var = 0.0f;
	for (int i = 0; i < nTests; i++) {
		s[i] = networks[0]->evaluateOnCloseness( X, Y);
		avg += s[i];
	}
	avg /= (float)nTests;
	for (int i = 0; i < nTests; i++) {
		var += powf(s[i] - avg, 2.0f);
	}
	var /= (float)nTests;
	std::cout << "var: " << var << std::endl;
	std::cout << "avg: " << avg << std::endl;

	delete[] X;
	delete[] Y;
	delete[] s;
}


void Population::step() {
	
	float* X = new float[networks[0]->inputSize];
	float* Y = new float[networks[0]->outputSize];
	

	for (int i = 0; i < nSpecimens; i++) {
		networks[i]->mutate();
		rawScores[i] = evaluateNetOnDataset(networks[i]);
		//rawScores[i] = -networks[i]->evaluateOnCloseness(X, Y);
	}



	rankArray(rawScores.data(), fitnesses.data(), nSpecimens);

	// logging scores. monitoring only, can be disabled.
	if (true) {
	
		float maxScore = -1000000.0f;
		int maxScoreID = 0;
		for (int i = 0; i < nSpecimens; i++) {

			if (rawScores[i] > maxScore) {
				maxScore = rawScores[i];
				maxScoreID = i;
			}
		}
		float avgavgf = 0.0f;
		for (float f : rawScores) avgavgf += f;
		avgavgf /= (float) nSpecimens;
		float bestC = networks[maxScoreID]->evaluateOnCloseness(X, Y);

		std::cout << "At generation " << evolutionStep
		<< ", max score = " << maxScore
		<< ", average score per specimen = " << avgavgf << ".\n"
		<< "SF(p*) " <<
		bestC << std::endl;		
	}

	delete[] X;
	delete[] Y;

	createOffsprings();
}


Network* Population::createChild(PhylogeneticNode* primaryParent) {

	// Reminder: when this function is called, the max number of parent is > 1.

	std::vector<int> parents;
	parents.push_back(primaryParent->networkIndice);
	
	std::vector<float> rawWeights;
	rawWeights.resize(nParents);

	// fill parents, and weights are initialized with a function of the phylogenetic distance.
	{
		PhylogeneticNode* previousRoot = primaryParent;
		PhylogeneticNode* root = primaryParent->parent;
		bool listFilled = false;

		auto distanceValue = [] (float d) 
		{
			return powf(1.0f+d, -0.6f);
		};

		for (int depth = 1; depth < MAX_MATING_DEPTH; depth++) {

			if (depth <= CONSANGUINITY_DISTANCE) 
			{
				previousRoot = root;
				root = root->parent;
				continue;
			}

			float w = distanceValue((float) depth);
			std::fill(rawWeights.begin() + (int)parents.size(), rawWeights.end(), w);

			int nC = (int)root->children.size();

			// we start at a random indice in the root children, so that if the list is filled before the end of this
			// loop iteration it is not always the same networks that are selected.
			int i0 = INT_0X(nC); 

			for (int i = 0; i < nC; i++) {
				int id = (i + i0) % nC;
				if (root->children[id] != previousRoot) {
					if (!root->children[id]->addToList(parents, depth-1)) {
						listFilled = true;
						break;
					}
				}
			}

			// parents.size() == nParents can happen if there were exactly as many parents 
			// as there was room left in the array. This does not set listFilled = true.
			if (listFilled || parents.size() == nParents) break; 

			previousRoot = root;
			root = root->parent;
		}
	}

	if (parents.size() == 1) {
		return new Network(networks[parents[0]]);
	}

	rawWeights[0] = 1.0f;
	float f0 = fitnesses[primaryParent->networkIndice];

	std::vector<Network*> parentNetworks;

#ifdef SPARSE_MUT_AND_COMB
	parentNetworks.push_back(networks[parents[0]]);
	for (int i = 1; i < parents.size(); i++) {
		if (fitnesses[parents[i]] > f0) {
			parentNetworks.push_back(networks[parents[i]]);
		}
	}
	if (parentNetworks.size() == 1) {
		return new Network(networks[parents[0]]);
	}
#else
	parentNetworks.resize(parents.size());
	for (int i = 0; i < parents.size(); i++) {
		parentNetworks[i] = networks[parents[i]];
	}
	for (int i = 1; i < parents.size(); i++) {
		rawWeights[i] *= (fitnesses[parents[i]] - f0); // TODO better
	}
#endif
	return Network::combine(parentNetworks, rawWeights.data());
}


void Population::createOffsprings() {
	uint64_t start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

	float* phase1Probabilities = new float[nSpecimens];
	float* phase2Probabilities = new float[nSpecimens];
	Network** tempNetworks = new Network* [nSpecimens];

	float fMax = -10000.0f;
	for (int i = 0; i < nSpecimens; i++) {
		if (fitnesses[i] > fMax) {
			fMax = fitnesses[i];
			fittestSpecimen = i;
		}
	}

	float normalizationFactor = 1.0f / (fMax - selectionPressure.first);
	for (int i = 0; i < nSpecimens; i++) {
		phase1Probabilities[i] = (fitnesses[i] - selectionPressure.first)*normalizationFactor;
		if (fitnesses[i] < selectionPressure.second) phase2Probabilities[i] = 0.0f;
		else phase2Probabilities[i] = fitnesses[i] - selectionPressure.second;
	}

	

	int parentPhylogeneticTreeID = ((evolutionStep-1) % MAX_MATING_DEPTH) * nSpecimens;
	int phylogeneticTreeID = (evolutionStep % MAX_MATING_DEPTH) * nSpecimens;
	int nReconductedSpecimens = 0;

	// lambda just not to write this twice.
	auto doEverything = [&](int parentID, int childID) {

		PhylogeneticNode* childNode = &phylogeneticTree[phylogeneticTreeID + childID];
		childNode->children.resize(0);
		childNode->parent = &phylogeneticTree[parentPhylogeneticTreeID + parentID];
		childNode->networkIndice = childID;
		childNode->parent->children.push_back(childNode);

		if (evolutionStep < MAX_MATING_DEPTH || nParents == 1) {
			tempNetworks[childID] = new Network(networks[parentID]);
		}
		else {
			tempNetworks[childID] = createChild(childNode->parent);
		}

		return;
	};

	
	for (int i = 0; i < nSpecimens; i++) {
		if (UNIFORM_01 < phase1Probabilities[i]) {
			if (i == fittestSpecimen) fittestSpecimen = nReconductedSpecimens; // happens once and only once
			doEverything(i, nReconductedSpecimens);
			nReconductedSpecimens++;
		}
	}
	//std::cout << "reconducted fraction : " << (float)nReconductedSpecimens / (float)nSpecimens << std::endl;

	// Compute probabilities for roulette wheel selection.
	float invProbaSum = 0.0f;
	for (int i = 0; i < nSpecimens; i++) {
		invProbaSum += phase2Probabilities[i];
	}
	invProbaSum = 1.0f / invProbaSum;

	phase2Probabilities[0] = phase2Probabilities[0] * invProbaSum;
	for (int i = 1; i < nSpecimens; i++) {
		phase2Probabilities[i] = phase2Probabilities[i - 1] + phase2Probabilities[i] * invProbaSum;
	}

	int parentID;
	for (int i = nReconductedSpecimens; i < nSpecimens; i++) {
		parentID = binarySearch(phase2Probabilities, UNIFORM_01, nSpecimens);
		doEverything(parentID, i);
	}


	// Clean up and update
	for (int i = 0; i < nSpecimens; i++) {
		delete networks[i];
	}

	for (int i = 0; i < nSpecimens; i++) {
		networks[i] = tempNetworks[i];
	}

	// d is used because for the MAX_MATING_DEPTH-1 first steps we cant traverse the deeper layers
	// of the tree. 2 is the stable value.
	int d = 2 + std::max(0, MAX_MATING_DEPTH - evolutionStep - 1);
	for (int i = parentPhylogeneticTreeID; i < parentPhylogeneticTreeID + nSpecimens; i++) {
		if (phylogeneticTree[i].children.size() == 0) {
			phylogeneticTree[i].erase(d);
		}
	}

	delete[] tempNetworks;
	delete[] phase1Probabilities;
	delete[] phase2Probabilities;
	
	evolutionStep++;

	uint64_t stop = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	//std::cout << "Offspring creation took " << stop - start << " ms." << std::endl;
}
