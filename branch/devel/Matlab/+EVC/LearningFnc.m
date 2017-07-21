classdef LearningFnc < EVC.EVCFnc
    
    % this class implements DDM-specific functions
    % most of the functions may reference an EVCDDM instance

    properties
       % output                               % fnc handle{} -> points to dynamic output value 
    end
    
    properties (Constant)
        SIMPLE_SALIENCY_RL = 1;             % learns about saliency based on error feedback
        SIMPLE_STATE_RL = 2;                % learns about any state representation based of experience of that representation
        FORAGING_REWARDS = 3;
        
        % holds amount of required parameters for each function 
        paramReqLF = [1, ...  SIMPLE_SALIENCY_RL: 1) learning rate
                      1, ...   SIMPLE_STATE_RL: 1) learning rate
                      1 ...    FORAGING_REWARDS: 1) learning rate ? (maybe not necessary)
                     ];
                
        % holds amount of required dynamic input parameters for each function
        inputReqLF = [2, ...  SIMPLE_SALIENCY_RL: 1) actual outcome 2) expected outcome
                      1, ...  SIMPLE_STATE_RL: 1) learned value
                      1 ...  FORAGING_REWARDS: 1) actual outcome
                     ];
    end
    
    methods
        
        function this = LearningFnc(type, params, varargin)
          
          % call superclass constructor to create an EVC function instance
          this = this@EVC.EVCFnc(type, params, 'paramReq', 'paramReqLF');
         
          [inputVal EVCM] = this.extractInput(varargin);
          this.input = inputVal;
          this.EVCModel = EVCM;

        end
        
    end
    
    methods (Access = public)
        
        % calculates output value dependent on specified function type
        function out = getVal(this, varargin)

            [inputVal, EVCM] = this.extractInput(varargin);
            
            % general learning functions
            
            if(this.type == this.SIMPLE_SALIENCY_RL)
               out = simpleSaliencyRL(this, EVCM); 
            end
            
            if(this.type == this.SIMPLE_STATE_RL)
               out = simpleStateRL(this, EVCM); 
            end
            
            if(this.type == this.FORAGING_REWARDS)
               out = setForagingRewards(this, EVCM); 
            end
            
            % implementation-specific learning functions may need an
            % additional verification of EVCM class (e.g. EVCDDM)
            
        end
       
    end
    
    methods (Access = private)
        
        %% function definitions
        
        % STIMBIAS: returns a response bias for the specified interval (specified
        % in this.params) given 
        function out = simpleSaliencyRL(this, EVCM)
            
            %% learn about automatic processing bias based on outcome differences between actual and expected state
            
            % EVCFnc parameters
            feedback = this.input{1}() - this.input{2}(); % actual error prob - expected error prob
            learningRate = this.params{1};
            
            % parameters retrieved from EVC model
            stateIdx = EVCM.getExpectedStateIdx();
            stimSalience = EVCM.State.ExpectedSpace(stateIdx).stimSalience();
            currTypeWeight = EVCM.getCurrentTypeWeight();

            normSalience = stimSalience / sum(stimSalience);
            
            stimRespMap = EVCM.State.ExpectedSpace(stateIdx).getStimRespMap();

            % reduce stimulus-response-mapping to one dimension
            stimRespWeights = stimRespMap;
            stimRespWeights(:,2) = stimRespWeights(:,2) * -1;    % val response 2 < 0 < val response 1
            stimRespWeights = transpose(sum(stimRespWeights,2)); % represents how much a given stimulus pushes towards one response (due to stimulus-response mapping) irrespective of it's salience
            
            normSalience = normSalience + stimRespWeights * learningRate * (-1) * feedback * currTypeWeight;
            normSalience = max(0.00001,normSalience);
            normSalience = normSalience / sum(normSalience);
            normSalience = min(0.99, normSalience);
            normSalience = max(0.01, normSalience);
            EVCM.State.ExpectedSpace(stateIdx).setSaliency(normSalience);
            
            %EVC.HelperFnc.disp(source, 'actual prop2', actualProp2, 'expected prop2', expectedProp2, 'old expected saliency', stimSalience, 'new expected saliency', EVCM.State.ExpectedSpace(stateIdx).stimSalience); % DIAGNOSIS
           
            out = normSalience;
        end
        
        function out = simpleStateRL(this, EVCM)
            
            % EVCFnc parameters
            feedback = this.input{1}();
            learningRate = this.params{1};
            
            % parameters retrieved from EVC model
            stateIdx = EVCM.getExpectedStateIdx();

            
            EVCM.State.Expected = EVCM.State.ExpectedSpace(stateIdx);
            %% learn about reward contingencies & stimulus response mappings
            
            EVCM.State.ExpectedSpace(stateIdx).outcomeValues = EVCM.State.Expected.outcomeValues + [1-feedback feedback] .* (EVCM.State.Actual.outcomeValues - EVCM.State.ExpectedSpace(stateIdx).outcomeValues) * learningRate;
            
            currentTaskSet = EVCM.currentTaskSet(1);
            EVCM.State.ExpectedSpace(stateIdx).stimRespMap = EVCM.State.ExpectedSpace(stateIdx).stimRespMap + repmat([1-feedback feedback],size(currentTaskSet,1),1) .* (currentTaskSet - EVCM.State.ExpectedSpace(stateIdx).stimRespMap) * learningRate;
            
            EVCM.State.Expected = EVCM.State.ExpectedSpace(stateIdx);
            
            out(1).learned = EVCM.State.ExpectedSpace(stateIdx).outcomeValues;
            out(2).learned = EVCM.State.ExpectedSpace(stateIdx).stimRespMap;
        end
        
        function out = setForagingRewards(this, EVCM)
            
            % this value is 1 if the agent choose to harvest current patch (hit the upper
            % threshold)
            harvestChoice = ~round(this.input{1}());
            
            % these are the previous trial rewards associated with
            % harvesting and switching
            oldHarvestReward = EVCM.State.Actual.outcomeValues(1);
            oldSwitchReward = EVCM.State.Actual.outcomeValues(2);
            
            % calculate new rewards here:
            startVols = (30:15:150)*1;
            %disp(['TRIAL' num2str(EVCM.State.TaskEnv.CurrTrialIdx)]);
            %disp(['harvest choice was ' num2str(harvestChoice) '. Old reward was ' num2str(oldHarvestReward)]);
            if(~harvestChoice)
                newHarvestReward = randsample(startVols, 1);
            else
                p = log10(oldHarvestReward/0.060233);
                newHarvestReward = 10^(p-.06)*0.060233;
                newHarvestReward = max(newHarvestReward, 6);
            end
            newSwitchReward = mean(startVols);
            %disp(['New reward is ' num2str(newHarvestReward)]);
            
            % feed new calculated values to EVC model
            if(EVCM.State.TaskEnv.CurrTrialIdx < length(EVCM.State.TaskEnv.Sequence))
                EVCM.State.TaskEnv.Sequence(EVCM.State.TaskEnv.CurrTrialIdx+1).outcomeValues = [newHarvestReward newSwitchReward];
            end
            stateIdx = EVCM.getExpectedStateIdx();
            EVCM.State.ExpectedSpace(stateIdx).outcomeValues = [newHarvestReward newSwitchReward];
            EVCM.State.Expected = EVCM.State.ExpectedSpace(stateIdx);
            
            out(1).learned = newHarvestReward;
            out(2).learned = newSwitchReward;
        end
    end
    
end