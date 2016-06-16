classdef EVCMSDDM < EVC.EVCModel
   
    properties

        MSDDMProcesses            % DDMproc[]: description of DDM parametrization
        
    end
    
    
    methods 
       
       % constructor
       function this = EVCMSDDM(CtrlSignals, TaskEnv, MSDDMProcesses)
          
          % call superclass constructor
          % note: this function automatically calls setActualState()

          this = this@EVC.EVCModel(CtrlSignals, TaskEnv);
          
          % reference model to DDM processes
          this.MSDDMProcesses = MSDDMProcesses;
          for i = 1:length(this.MSDDMProcesses)
             this.MSDDMProcesses(i).input.EVCModel = this;
          end

          
       end

        
       % calculates probabilities & RTs for each possible outcome depending on the current
        function [performance]= simulateOutcomes(this)
            [ST,I] = dbstack; % track function name for debugging functions
            source = ST.name;
            
            import EVC.MSDDM.*;
            
            % parameterize MSDDM
            stages = MSDDMProc.getStages(this.MSDDMProcesses);
            
            numSignals = size(this.getIntensities(),2);
            drift = zeros(numSignals,length(stages));
            noise = zeros(numSignals,length(stages));
            thresh = zeros(numSignals,length(stages));
            deadlines = zeros(numSignals,length(stages));
            x0 = zeros(numSignals,1);
            T0 = zeros(numSignals,1);
            
            if(this.useActualState)
                stateType = MSDDMProc.ACTUAL_STATE;
            else
                stateType = MSDDMProc.EXPECTED_STATE;
            end
            
            allowedTypes = [stateType MSDDMProc.DEFAULT MSDDMProc.CONTROL stateType];
            
            % specify x0 and T0 from first stage
            biasProcesses = MSDDMProc.filterProcess(this.MSDDMProcesses, 'stage', 1, 'type', allowedTypes, 'DDMProxy', MSDDMProc.BIAS);
            for i = 1:length(biasProcesses)
               x0(:) = x0(:) + biasProcesses.getVal();
            end
            
            T0Processes = MSDDMProc.filterProcess(this.MSDDMProcesses, 'stage', 1, 'type', allowedTypes, 'DDMProxy', MSDDMProc.T0);
            for i = 1:length(T0Processes)
               T0(:) = T0(:) + T0Processes.getVal();
            end
            
            % loop through all stages and specify MSDDM parameters for each
            % stage (except x0 and T0)
            for stageIdx = 1:length(stages)
               
                currStage = stages(stageIdx);
                
                % add drift rates
                driftProcesses = MSDDMProc.filterProcess(this.MSDDMProcesses, 'stage', currStage, 'type', allowedTypes, 'DDMProxy', MSDDMProc.DRIFT);
                for i = 1:length(driftProcesses)
                    drift(:,stageIdx) = drift(:,stageIdx) + driftProcesses(i).getVal()';
                end
                
                % add thresholds
                threshProcesses = MSDDMProc.filterProcess(this.MSDDMProcesses, 'stage', currStage, 'type', allowedTypes, 'DDMProxy', MSDDMProc.THRESH);
                for i = 1:length(threshProcesses)
                    thresh(:,stageIdx) = thresh(:,stageIdx) + threshProcesses(i).getVal()';
                end
                
                % add diffusion components
                noiseProcesses = MSDDMProc.filterProcess(this.MSDDMProcesses, 'stage', currStage, 'type', allowedTypes, 'DDMProxy', MSDDMProc.NOISE);
                for i = 1:length(noiseProcesses)
                    noise(:,stageIdx) = noise(:,stageIdx) + noiseProcesses(i).getVal()';
                end
                
                allProcesses = MSDDMProc.filterProcess(this.MSDDMProcesses, 'stage', currStage, 'type', allowedTypes);
                
                % specify duration
                if(stageIdx < length(stages))
                    longestDeadline = zeros(numSignals,1);      % take maximum deadline across all processes of stage
                    for i = 1:length(allProcesses)
                        if(length(allProcesses(i).getDuration()) == 1)
                            longestDeadline(:) = max(allProcesses(i).getDuration()', longestDeadline);
                        else
                            if(length(allProcesses(i).getDuration()) ~= longestDeadline)
                                warning('Number of MSDDM process durations and number of control signals doesn''t match.');
                            end
                            longestDeadline(:) = max([allProcesses(i).getDuration()', longestDeadline], 2);
                        end
                    end
                    deadlines(:,stageIdx+1) = deadlines(:,stageIdx) + longestDeadline;
                end
                
            end
            
            deadlines = deadlines+ repmat(T0,1,size(deadlines,2));
                        
            %EVC.HelperFnc.disp(source, 'drift', drift, 'thresh', thresh, 'bias', x0, 'noise', noise, 'T0', T0, 'deadlines', deadlines); % DIAGNOSIS

            % call MS DDM  
            aRT = zeros(size(drift,1), 1);
            aER = zeros(size(drift,1), 1);
            for i = 1:size(drift,1)
            
                [aRT(i), aER(i)]= multi_stage_ddm_metrics(drift(i,:),noise(i,:), deadlines(i,:), thresh(i,:), x0(i), 1);
            end
            
%             if(~this.useActualState)
%                 disp('---');
%                disp('drift');
%                disp(drift);
%                disp('noise');
%                disp(noise);
%                disp('deadlines');
%                disp(deadlines);
%                disp('thresh');
%                disp(thresh);
%                disp('x0');
%                disp(x0);
%                disp('RT');
%                disp(aRT);
%                disp('ER');
%                disp(aER);
%             end
            
            % check for weird outcomes
%             if(drift(end) == 0)
%                 warning('zero drift rate at last stage.');
%                 tmpER(drift==0) = x0;
%                 tmpFullRT(drift==0) = 1e+12;
%             end
            
            if(~isreal(aER))
                disp('drift, bias, thresh');
                disp([drift bias ddmp.z]);
               error('Something imaginary bad happened.');
            end

            % calculate outcomes
            probs(1,:) = (1-aER);
            probs(2,:) = aER;
            RTs = [aRT'; aRT'];                     % use only mean RT for now
            
            % handle binary errors
            if(this.binaryErrors && this.useActualState)
               number = rand;
               if(number < probs(1))
                   probs(1) = 1;
               else
                   probs(1) = 0;
               end
               probs(2) = 1 - probs(1);
            end

            performance.probs = probs;
            performance.RTs = RTs;
            
            EVC.HelperFnc.disp(source, 'probs', probs, 'RTs', RTs); % DIAGNOSIS
            
        end
        

        
        % logs trial outcomes for current control signals
        function log(this, varargin)
            
            % call superclass log function
            if(length(varargin)>0)
                superclass_args = varargin;
                log@EVC.EVCModel(this, superclass_args{:})
            else 
                log@EVC.EVCModel(this);
            end
            
            % log specific parameters from EVCDDM
            
            % - log actual error rate
            expectedProbs = this.State.Expected.performance.probs;
            actualProbs = this.State.Actual.performance.probs;
            
            if(this.State.Actual.outcomeValues(2) > this.State.Actual.outcomeValues(1)) % if second option is defined as the correct outcome, then 1st column must provide error probability
                ERcol = 1;
            else
                ERcol = 2;
            end
            expectedER = expectedProbs(ERcol);
            actualER = actualProbs(ERcol);
            this.Log.ERs(this.Log.LogCount,:) = [expectedER actualER];
 
            % - log control parameter
            controlProc = EVC.MSDDM.MSDDMProc.filterProcess(this.MSDDMProcesses, 'type', EVC.MSDDM.MSDDMProc.CONTROL);
            proxyArray = NaN(1,length(controlProc));
            valArray = NaN(1,length(controlProc));
            for i = 1:length(controlProc)
               proxyArray(i) = controlProc(i).DDMProxy;
               valArray(i) = controlProc(i).getVal();
            end
            this.Log.ControlParamType(this.Log.LogCount,:) = proxyArray;
            this.Log.ControlParamVal(this.Log.LogCount,:) = valArray;
            
            % - log expected state parameter
            expectedProc = EVC.MSDDM.MSDDMProc.filterProcess(this.MSDDMProcesses, 'type', EVC.MSDDM.MSDDMProc.EXPECTED_STATE);
            this.Log.ExpectedStateParam(this.Log.LogCount,:) = expectedProc.getVal();
            
            
            % - log actual state parameter
            actualProc = EVC.MSDDM.MSDDMProc.filterProcess(this.MSDDMProcesses, 'type', EVC.MSDDM.MSDDMProc.ACTUAL_STATE);
            this.Log.ActualStateParam(this.Log.LogCount,:) = actualProc.getVal();
            
            
        end
        
    end

    
end