function fitVal = MSDDM_GrattonSimple_wrapper(costParam1, costParam2, defaultThresh, driftEfficacy, automaticDuration)

    import Simulations.*;

    EVCSim = MSDDM_GrattonSimple();
    EVCSim.printResults = 0;
    EVCSim.nSubj = 1;

    % set parameters
    EVCSim.defaultCostFnc.params{1} = costParam1; 
    EVCSim.defaultCostFnc.params{2} = costParam2; 
    EVCSim.defaultDDMParams.thresh = defaultThresh; 
    EVCSim.defaultControlMappingFnc.params{2} = driftEfficacy;
    EVCSim.defaultAutomaticProcess.duration.params{1} = automaticDuration;

    EVCSim.wrapper = 1;
    EVCSim.run();
    
    fitVal = EVCSim.getOptimizationCriterion();

end