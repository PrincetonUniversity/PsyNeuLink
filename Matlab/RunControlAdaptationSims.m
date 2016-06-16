%% EVC CONTROL ADAPTATION SIMULATIONS

%%
clc;
clear all;
import Simulations.*;

%% DDM: simulates trial-by-trial adjustments to errors (see Laming, 1968)
EVCSim = DDM_Laming();
EVCSim.printResults = 0;
EVCSim.plotSum = 1;
EVCSim.run();

%% MSDDM: simulates trial-by-trial adjustments to errors (see Laming, 1968)
EVCSim = MSDDM_Laming();
EVCSim.printResults = 0;
EVCSim.nSubj = 1;
EVCSim.plotSum = 1;
EVCSim.run();

%% DDM: simulates sequential control adjustments in response to conflict (see Gratton, 1992)
EVCSim = DDM_GrattonSimple();
EVCSim.printResults = 1;
EVCSim.nSubj = 20;
EVCSim.plotSum = 0;
EVCSim.run();

%% MSDDM: simulates sequential control adjustments in response to conflict (see Gratton, 1992)
EVCSim = MSDDM_GrattonSimple();
EVCSim.printResults = 2;
EVCSim.nSubj = 10;
EVCSim.nTrials = 100;
EVCSim.plotSum = 1;
EVCSim.run();

%% simulates sequential control adjustments in response to conflict (see Sternbergen, 2015)
EVCSim = DDM_Steenbergen2015();
EVCSim.printResults = 0;
EVCSim.nSubj = 1;
EVCSim.plotSum = 1;
EVCSim.run();

%% runs a simulation with reward manipulations of the current task
EVCSim = DDM_RewardPerturb();
EVCSim.printResults = 0;
EVCSim.plotSum = 1;
EVCSim.run();

%% runs a conflict task simulation with reward manipulation  (according to Padmala & Pessoa, 2011)
EVCSim = DDM_Padmala2011();
EVCSim.printResults = 0;
EVCSim.plotSum = 1;
EVCSim.run();
