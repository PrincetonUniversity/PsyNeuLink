%% EVC SANITY CHECKS

%%
clc
clear all;
import Simulations.*;

results = 1;

%% runs a basic EVC DDM implementation to check general simulation functionality
EVCSim = DDMSim();
EVCSim.printResults = results;
EVCSim.run();

%% runs a basic EVC MSDDM implementation to check general simulation functionality
EVCSim = MSDDMSim();
EVCSim.printResults = results;
EVCSim.run();

%% runs a simulation with systematic reward manipulations of the current task
EVCSim = DDMRewardSanity();
EVCSim.printResults = results;
EVCSim.nSubj = 10;
EVCSim.run();

%% runs a simulation with difficulty manipulations of the current task
EVCSim = DDMDifficultySanity();
EVCSim.printResults = results;
EVCSim.run();