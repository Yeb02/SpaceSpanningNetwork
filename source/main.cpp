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
#include <algorithm>
#include "Population.h"
#include "Random.h"

#define LOGV(v) for (const auto e : v) {cout << std::setprecision(2)<< e << " ";}; cout << "\n"
#define LOG(x) cout << x << endl;

using namespace std;



int main()
{
    LOG("Seed : " << seed);

    const int IN_SIZE = 5;
    const int OUT_SIZE = 25;


    // Gradient based method:

    if (false) {
        float lr = .11f;

        Network n(IN_SIZE, OUT_SIZE);

        float* X = new float[IN_SIZE];
        float* Y = new float[OUT_SIZE];

        float* datasetY = new float[OUT_SIZE * N_YS];
        for (int k = 0; k < OUT_SIZE * N_YS; k++)
        {
            datasetY[k] = clamp(NORMAL_01 * .3f, -1.0f, 1.0f);
        }
        float* datasetX = new float[IN_SIZE * N_YS];
        for (int k = 0; k < IN_SIZE * N_YS; k++)
        {
            datasetX[k] = NORMAL_01;
        }


        for (int i = 0; i < 1000; i++)
        {
            // before GD to evaluate initial Net
            float bestC = n.evaluateOnCloseness(X, Y);

            float l = 0.0f;
            for (int j = 0; j < N_YS; j++) {
                l += n.forward(&datasetX[j * IN_SIZE], &datasetY[j * OUT_SIZE], true);
                //n.updateParams(lr*.01f);
            }
            n.updateParams(lr / (float)N_YS);


            std::cout << "Epoch " << i
                << " loss " << l / (float)N_YS
                << " SF(p) " <<
                bestC << std::endl;
        }

        delete[] X;
        delete[] Y;

        return 0;
    }

    // Population based method :


    int nSpecimens = 512; //16 -> 512 in most cases
    int nSteps = 10000;

    PopulationEvolutionParameters params;
    params.selectionPressure = { -3.0f, .2f}; 
    params.nParents = 50;

    Population population(IN_SIZE, OUT_SIZE, nSpecimens);
    population.setEvolutionParameters(params); 

    population.test();

    for (int i = 0; i < nSteps; i++) {

        //float c = std::max(.0F, 1.0f - (float)i / 20.0f);
        //params.selectionPressure.first = (1.0f - c) * (1.5f * sinf((float)i / 3.0f) - 1.5f) + c * -1.0f;
        //LOG("Pressure was : " << params.selectionPressure.first);
        //population.setEvolutionParameters(params); // parameters can be changed at each step.

        population.step();

        if ((i + 1) % 30 == 0) {
            population.saveFittestSpecimen();
        }
    }

    return 0;
}
