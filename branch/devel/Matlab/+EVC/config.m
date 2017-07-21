%% pathway file for EVC simulation (running/debugging)


diagnosis = 0;                                                              % if turned on, displays diagnostic outputs

% specific function diagnosis
Diagnosis.EVCModel.getEVC = 0;
Diagnosis.EVCModel.getOptSignals = 0;
Diagnosis.EVCModel.currentTaskSet = 0;
Diagnosis.EVCModel.writeLogFile = 0;
Diagnosis.EVCModel.getBOLD = 0;
Diagnosis.EVCDDM.setActualState = 0;                                    
Diagnosis.EVCDDM.updateExpectedState = 0;                                
Diagnosis.EVCDDM.getControlHandle = 0;
Diagnosis.EVCDDM.simulateOutcomes = 1;
Diagnosis.EVCDDM.StimSalRespMap = 0;
Diagnosis.EVCDDM.setStateParams = 0;
Diagnosis.EVCDDM.getStimSalRespMap = 0;
Diagnosis.EVCDDM.getAutomaticBias = 0;

warnings = false;                                                            % if turned on, display warnings
