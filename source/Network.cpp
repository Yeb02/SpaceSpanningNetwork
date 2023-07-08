#pragma once

#include "Network.h"


Network::Network(int inputSize, int outputSize) :
	inputSize(inputSize), outputSize(outputSize)
{
	nLayers = 1; // 1 if no hidden

	sizes.push_back(inputSize);
	for (int i = 0; i < nLayers - 1; i++) { // TODO
		sizes.push_back(10);
	}
	sizes.push_back(outputSize);

	// init Ws and Bs
	for (int i = 0; i < nLayers; i++)
	{
		float f = powf((float)sizes[i], -.5f);
		int sW = sizes[i] * sizes[i + 1];
		Ws.emplace_back(new float[sW]);
		for (int j = 0; j < sW; j++) {
			Ws[i][j] = NORMAL_01 * f;
		}

		int sB = sizes[i + 1];
		Bs.emplace_back(new float[sizes[i + 1]]);
		for (int j = 0; j < sizes[i + 1]; j++) {
			Bs[i][j] = NORMAL_01;
		}
	}


	output = new float[outputSize];

	int activationS = 0;
	for (int i = 0; i < nLayers + 1; i++) {
		activationS += sizes[i];
	}
	activations = std::make_unique<float[]>(activationS);
	delta = std::make_unique<float[]>(activationS - inputSize);


	WGrads.reserve(nLayers);
	BGrads.reserve(nLayers);

	for (int i = 0; i < nLayers; i++)
	{
		int sW = sizes[i] * sizes[i + 1];
		WGrads.emplace_back(new float[sW]);
		std::fill(WGrads[i].get(), WGrads[i].get() + sW, 0.0f);

		int sB = sizes[i + 1];
		BGrads.emplace_back(new float[sB]);
		std::fill(BGrads[i].get(), BGrads[i].get() + sB, 0.0f);
	}
}


float Network::forward(float* X, float* Y, bool accGrad)
{

	// forward
	float* prevActs = &activations[0];
	std::copy(X, X + inputSize, prevActs);
	float* currActs = &activations[inputSize];
	for (int i = 0; i < nLayers; i++) {
		for (int j = 0; j < sizes[i + 1]; j++) {
			currActs[j] = Bs[i][j];
		}

		int matID = 0;
		for (int j = 0; j < sizes[i + 1]; j++) {
			for (int k = 0; k < sizes[i]; k++) {
				currActs[j] += Ws[i][matID] * prevActs[k];
				matID++;
			}
		}

		for (int j = 0; j < sizes[i + 1]; j++) {
			currActs[j] = tanhf(currActs[j]); // Could be in the previous loop, but GPUisation.
		}

		prevActs = currActs;
		currActs = currActs + sizes[i + 1];
	}

	float loss = 0.0f;
	for (int i = 0; i < outputSize; i++) // euclidean distance loss
	{
		loss += powf(prevActs[i] - Y[i], 2.0f);
	}

	if (!accGrad) {
		return loss;
	}

	// backward. We will use tanh'(x) = 1 - tanh(x)², most stable and re-uses forward's calculations.
	float* prevDelta;
	float* currDelta = &delta[0];
	currActs = prevActs;
	prevActs = currActs - sizes[nLayers - 1];
	for (int i = 0; i < outputSize; i++) // euclidean distance loss
	{
		currDelta[i] = (currActs[i] - Y[i]) * (1.0f - currActs[i] * currActs[i]);
	}

	// w
	int matID = 0;
	for (int j = 0; j < sizes[nLayers]; j++) {
		for (int k = 0; k < sizes[nLayers - 1]; k++) {
			WGrads[nLayers - 1][matID] += currDelta[j] * prevActs[k];
			matID++;
		}
	}

	// b
	for (int j = 0; j < sizes[nLayers]; j++)
	{
		BGrads[nLayers - 1][j] += currDelta[j];
	}


	for (int i = nLayers - 1; i >= 1; i--) {
		prevDelta = currDelta;
		currDelta = currDelta + sizes[i + 1];
		currActs = prevActs; // Yes opposed to delta. I swear this makes sense.
		prevActs = currActs - sizes[i - 1];

		for (int j = 0; j < sizes[i]; j++) {
			currDelta[j] = 0.0f;
		}

		// Update deltas. This way of seeing the matmul avoids non-trivial indices induced by the transposition.
		matID = 0;
		for (int j = 0; j < sizes[i + 1]; j++) {
			for (int k = 0; k < sizes[i]; k++) {
				currDelta[k] += Ws[i][matID + k] * prevDelta[j];
			}
			matID += sizes[i];
		}
		for (int j = 0; j < sizes[i]; j++) {
			currDelta[j] *= (1.0f - currActs[j] * currActs[j]);
		}

		// w
		matID = 0;
		for (int j = 0; j < sizes[i]; j++) {
			for (int k = 0; k < sizes[i - 1]; k++) {
				WGrads[i - 1][matID] += currDelta[j] * prevActs[k];
				matID++;
			}
		}

		// b
		for (int j = 0; j < sizes[i]; j++)
		{
			BGrads[i - 1][j] += currDelta[j];
		}

	}

	return loss;
}


void Network::updateParams(float lr)
{
	for (int i = 0; i < nLayers; i++)
	{
		int sW = sizes[i] * sizes[i + 1];
		Ws.emplace_back(new float[sW]);
		for (int j = 0; j < sW; j++) {
			Ws[i][j] = WGrads[i][j] * lr;
		}
		std::fill(WGrads[i].get(), WGrads[i].get() + sW, 0.0f);

		int sB = sizes[i + 1];
		Bs.emplace_back(new float[sB]);
		for (int j = 0; j < sB; j++) {
			Bs[i][j] = BGrads[i][j] * lr;
		}
		std::fill(BGrads[i].get(), BGrads[i].get() + sB, 0.0f);
	}
}


Network::Network(Network* n) : Network(n->inputSize, n->outputSize)
{
	for (int i = 0; i < nLayers; i++)
	{
		int sW = sizes[i] * sizes[i + 1];
		std::copy(n->Ws[i].get(), n->Ws[i].get() + sW, Ws[i].get());

		int sB = sizes[i + 1];
		std::copy(n->Bs[i].get(), n->Bs[i].get() + sB, Bs[i].get());
	}
}


void Network::mutate()
{
	constexpr float f = .3f;
	constexpr float p = .1f;
	int nMutations;

	for (int i = 0; i < nLayers; i++)
	{ 
		int sW = sizes[i] * sizes[i + 1];
		int sB = sizes[i + 1];

#ifdef SPARSE_MUT_AND_COMB
		SET_BINOMIAL(sW, p);
		nMutations = BINOMIAL;
		for (int j = 0; j < nMutations; j++) {
			Ws[i][INT_0X(sW)] += NORMAL_01;
		}
		SET_BINOMIAL(sB, p);
		nMutations = BINOMIAL;
		for (int j = 0; j < nMutations; j++) {
			Bs[i][INT_0X(sB)] += NORMAL_01;
		}
#else
		// Global
		for (int j = 0; j < sB; j++) {
			Bs[i][j] += f * NORMAL_01;
		}
		for (int j = 0; j < sW; j++) {
			Ws[i][j] += f * NORMAL_01;
		}
#endif
	}
}


Network* Network::combine(std::vector<Network*>& parents, float* weights) {
	Network* child = new Network(parents[0]);

	int nParents = (int)parents.size();

	float** mats = new float* [nParents];


	// accumulates in mats[0]
	auto addMatrices = [mats, weights, nParents](int s)
	{
		for (int j = 0; j < s; j++) {
			mats[0][j] *= weights[0];
		}
		for (int i = 1; i < nParents; i++) {
			for (int j = 0; j < s; j++) {
				mats[0][j] += weights[i] * mats[i][j];
			}
		}
	};

	// weights
#ifndef SPARSE_MUT_AND_COMB
	{
		float s_positive = 0.0f, s_negative = 0.0f;
		for (int j = 1; j < nParents; j++) {
			if (weights[j] > 0) {
				s_positive += weights[j];
			}
			else {
				s_negative -= weights[j];
			}
		}
		float s_max = std::max(s_positive, s_negative);
		float s_f = .4f / s_max;
		weights[0] = 1 - (s_positive - s_negative) * s_f;
		for (int j = 1; j < nParents; j++) {
			weights[j] *= s_f;
		}
	}
#endif


	for (int i = 0; i < child->nLayers; i++)
	{
		int sW = child->sizes[i] * child->sizes[i + 1];
		int sB = child->sizes[i + 1];

#ifdef SPARSE_MUT_AND_COMB
		for (int j = 0; j < sW; j++) {
			int pID = INT_0X(nParents);
			child->Ws[i][j] = parents[pID]->Ws[i][j];
		}
		for (int j = 0; j < sB; j++) {
			int pID = INT_0X(nParents);
			child->Bs[i][j] = parents[pID]->Bs[i][j];
		}
#else
		mats[0] = child->Ws[i].get();
		for (int j = 1; j < nParents; j++) {
			mats[j] = parents[j]->Ws[i].get();
		}
		addMatrices(sW);

		mats[0] = child->Bs[i].get();
		for (int j = 1; j < nParents; j++) {
			mats[j] = parents[j]->Bs[i].get();
		}
		addMatrices(sB);
#endif
	}
	
	return child;
}