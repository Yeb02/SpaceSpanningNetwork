#pragma once

#ifdef _DEBUG
// https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/control87-controlfp-control87-2?view=msvc-170
// These are incompatible with RocketSim that has many float errors, and should be commented when rocketsim.h and 
// .cpp are included in the project (so exclude them temporarily to use this feature).
#define _CRT_SECURE_NO_WARNINGS
#include <float.h>
unsigned int fp_control_state = _controlfp(_EM_UNDERFLOW | _EM_INEXACT, _MCW_EM);

#endif

#include <iostream>
#include "Population.h"
#include "Random.h"

#define LOGV(v) for (const auto e : v) {cout << std::setprecision(2)<< e << " ";}; cout << "\n"
#define LOG(x) cout << x << endl;

using namespace std;



int main()
{
    LOG("Seed : " << seed);

    int nSpecimens = 128; //16 -> 512 in most cases
    int nSteps = 10000;
    const int IN_SIZE = 10;
    const int OUT_SIZE = 50;

    PopulationEvolutionParameters params;
    params.selectionPressure = { -2.0f, .3f}; 
    params.useSameTrialInit = false; 
    params.rankingFitness = true;
    params.saturationFactor = .08f;
    params.regularizationFactor = .03f; 
    params.competitionFactor = .0f; 
    params.scoreBatchTransformation = NONE; // NONE recommended when useSameTrialInit = false
    params.nParents = 1;

    Population population(IN_SIZE, OUT_SIZE, nSpecimens);
    population.setEvolutionParameters(params); 



    
    for (int i = 0; i < nSteps; i++) {

        float c = std::max(.0F, 1.0f - (float)i / 20.0f);
        params.selectionPressure.first = (1.0f - c) * (1.5f * sinf((float)i / 3.0f) - 1.5f) + c * -1.0f;
        LOG("Pressure was : " << params.selectionPressure.first);
        population.setEvolutionParameters(params); // parameters can be changed at each step.

        population.step();

        if ((i + 1) % 30 == 0) {
            population.saveFittestSpecimen();
        }
    }

    return 0;
}
