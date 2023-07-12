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

    const int IN_SIZE = 1;
    const int OUT_SIZE = 2;


    // Gradient based method:

    if (true) {
        float lr = .01f;
        float regW = .03f;
        float regB = .1f;

        Network n(IN_SIZE, OUT_SIZE);

        float* X = new float[IN_SIZE];
        float* Y = new float[OUT_SIZE];

        int datasetSize = 2000;
        float* datasetY = new float[OUT_SIZE * datasetSize];
        float* datasetX = new float[IN_SIZE * datasetSize];

        // test to make sure the implementation of the network behaves as pytorch's
        if (false){
            X[0] = 1.0f;
            X[1] = 2.0f;

            float b0[4] = { -0.1836, -0.0931, -0.0471, -0.1128 };
            float w0[2 * 4] = { 0.1420, 1.1912,
            -0.5996, -1.1739,
                -0.0617, -0.0283,
                0.2927, -0.4975 };

            float b1[3] = { 0.0909,  0.1073, -0.1926 };
            float w1[3 * 4] = { 7.9108e-01,  4.7844e-01, -1.7395e-01,  4.2979e-02,
            1.0240e+00, -1.0854e-03,  1.2352e+00,  8.6315e-01,
            -4.4526e-01,  1.2464e+00,  1.0384e-02,  1.0814e+00 };

            for (int i = 0; i < 4; i++) {
                n.Bs[0][i] = b0[i];
                for (int j = 0; j < 2; j++) {
                    n.Ws[0][2 * i + j] = w0[2 * i + j];
                }
            }
            for (int i = 0; i < 3; i++) {
                n.Bs[1][i] = b1[i];
                for (int j = 0; j < 4; j++) {
                    n.Ws[1][4 * i + j] = w1[4 * i + j];
                }
            }

            n.forward(X, Y, NO_GRAD);
        }
        
        for (int i = 0; i < 1000; i++)
        {
            if (i % 10000 == 0) 
            {
                for (int k = 0; k < IN_SIZE * datasetSize; k++)
                {
                    datasetX[k] = NORMAL_01;
                }
                for (int k = 0; k < OUT_SIZE * datasetSize; k++)
                //for (int k = 0; k < IN_SIZE * datasetSize; k++)
                {
                    //datasetY[2*k] = cosf(10.0f * datasetX[k / 2]) * datasetX[k / 2] *.3f;
                    //datasetY[2*k + 1] = sinf(10.0f * datasetX[k / 2]) * datasetX[k / 2] *.3f;
                    datasetY[k] = clamp(NORMAL_01 * .3f, -1.0f, 1.0f);
                }
            }
            

            // before GD to evaluate initial Net
            float SF = n.evaluateOnCloseness(X, Y);
            std::cout << n.evaluateOnCloseness(X, Y)<< std::endl;
            std::cout << n.evaluateOnCloseness(X, Y)<< std::endl;
            std::cout << n.evaluateOnCloseness(X, Y)<< std::endl;

            float l = 0.0f;
            for (int j = 0; j < datasetSize; j++) {
                l += n.forward(&datasetX[j * IN_SIZE], &datasetY[j * OUT_SIZE], true);
                n.updateParams(lr, regW, regB); // inline
            }
            //n.updateParams(lr / (float)datasetSize, regW, regB);


            std::cout << "Epoch " << i
                << " loss " << l / (float)datasetSize
                << " SF(p) " <<
                SF << std::endl;


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
