%% EVC TASK SWITCHING SIMULATIONS

%%
clc;
clear all;
import Simulations.*;

%% DDM: simulates a blocked task switching paradigm (see Rogers & Monsell, 1995)
EVCSim = DDMTaskSwitchingBlocked();
model = EVCSim.initDebugging();
EVCSim.plotSum = 1;
EVCSim.run();

%% MSDDM: simulates a blocked task switching paradigm (see Rogers & Monsell, 1995)
EVCSim = MSDDMTaskSwitchingBlocked();
model = EVCSim.initDebugging();
EVCSim.plotSum = 1;
EVCSim.nSubj = 1;

EVCSim.run();

%% simulates a blocked task switching paradigm with one task being more rewarded than the other (see Umemoto & Holroyd, 2014)
EVCSim = DDM_TS_AsymmReward();
model = EVCSim.initDebugging();
EVCSim.plotSum = 1;
EVCSim.run();

%% simulates a voluntary task switching situation in which the reward structure of two tasks switches block by block
EVCSim = DDM_VTS_RewardBlocks();
model = EVCSim.initDebugging();
EVCSim.plotSum = 1;
EVCSim.run();

%% simulates a voluntary task switching situation in which the difficulty of the initially easier is increased continously in order to investigate the tention between conflict adjustment and avoidance
EVCSim = DDM_VTS_ConflictAvoidApproach();
model = EVCSim.initDebugging();
EVCSim.plotSum = 1;
EVCSim.run();

%% simulates a foraging task
clc
clear all;
import Simulations.*;

results = 1;
EVCSim = DDM_Foraging();
EVCSim.printResults = results;
EVCSim.run();