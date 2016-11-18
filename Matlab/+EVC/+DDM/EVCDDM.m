classdef EVCDDM < EVC.EVCModel
   
    properties

        DDMProcesses            % DDMproc[]: description of DDM parametrization
        
    end
    
    
    methods 
       
       % constructor
       function this = EVCDDM(CtrlSignals, TaskEnv, DDMProcesses)
          
          % call superclass constructor
          % note: this function automatically calls setActualState()

          this = this@EVC.EVCModel(CtrlSignals, TaskEnv);
          
          % reference model to DDM processes
          this.DDMProcesses = DDMProcesses;
          for i = 1:length(this.DDMProcesses)
             this.DDMProcesses(i).input.EVCModel = this;
          end

          
       end
       
       
      
 
        
       % calculates probabilities & RTs for each possible outcome depending on the current
        function [performance]= simulateOutcomes(this)
            [ST,I] = dbstack; % track function name for debugging functions
            source = ST.name;
            
            import EVC.DDM.*;
            
            drift = 0;
            bias = 0;
            ddmp.z = 0;
            ddmp.c = 0;
            ddmp.t0 = 0;
            
            if(this.useActualState)
                stateType = DDMProc.ACTUAL_STATE;
            else
                stateType = DDMProc.EXPECTED_STATE;
            end
            
            for i = 1:length(this.DDMProcesses)
               
                % check if ddm parameter is either control, default or
                % relevant state parameter
                if(ismember(DDMProc.DEFAULT,this.DDMProcesses(i).type) ...
                   || ismember(DDMProc.CONTROL,this.DDMProcesses(i).type) ...
                   || ismember(stateType,this.DDMProcesses(i).type))
                        
                    switch this.DDMProcesses(i).DDMProxy
                        case DDMProc.DRIFT
                             drift = drift + this.DDMProcesses(i).getVal();
                        case DDMProc.THRESH
                             ddmp.z = ddmp.z + this.DDMProcesses(i).getVal();
                        case DDMProc.BIAS
                             bias = bias + this.DDMProcesses(i).getVal();
                        case DDMProc.NOISE
                             ddmp.c = ddmp.c + this.DDMProcesses(i).getVal();
                        case DDMProc.t0
                             ddmp.t0 = ddmp.t0 + this.DDMProcesses(i).getVal();
                    end
                    
                end
            end
                        
            EVC.HelperFnc.disp(source, 'drift', drift, 'ddmp.z', ddmp.z, 'bias', bias, 'ddmp.c', ddmp.c); % DIAGNOSIS

            % call DDM
            %[tmpER,~,~,~,~,~,allFinalRTs_sepCDFs] = AS_ddmSimFRG(drift,bias,0,ddmp,0,0,null(1),[1], 1); % separate RT's
            [tmpER,~,~,tmpFullRT] = AS_ddmSimFRG_Mat(drift,bias,ddmp); % use only mean RT for now; tmpER represents probability of hitting bottom threshold
            
            
            % check for weird outcomes
            tmpER(drift==0) = bias;
            tmpFullRT(drift==0) = 1e+12;
            
            if(~isreal(tmpER))
                disp('drift, bias, thresh');
                disp([drift bias ddmp.z]);
               error('Something imaginary bad happened.');
            end

            % calculate outcomes
            probs(1,:) = (1-tmpER);
            probs(2,:) = tmpER;
            RTs = [tmpFullRT; tmpFullRT];                     % use only mean RT for now
            
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
            
            % TODO check this... can't update actual state when using binary errors because log()
            % function calls getEVC() again, so the binary probs would
            % change
%             if(this.useActualState)
%                %this.State.Actual.probs = probs;
%                %this.State.Actual.RTs = RTs; 
%             else
%                this.State.Expected.probs = probs;
%                this.State.Expected.RTs = RTs;  
%             end

            performance.probs = probs;
            performance.RTs = RTs;
            
            EVC.HelperFnc.disp(source, 'tmpER', tmpER, 'RTs', RTs); % DIAGNOSIS
            
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
            controlProc = EVC.DDM.DDMProc.filterProcess(this.DDMProcesses, EVC.DDM.DDMProc.CONTROL);
            proxyArray = NaN(1,length(controlProc));
            valArray = NaN(1,length(controlProc));
            for i = 1:length(controlProc)
               proxyArray(i) = controlProc(i).DDMProxy;
               valArray(i) = controlProc(i).getVal();
            end
            this.Log.ControlParamType(this.Log.LogCount,:) = proxyArray;
            this.Log.ControlParamVal(this.Log.LogCount,:) = valArray;
            
            % - log expected state parameter
            expectedProc = EVC.DDM.DDMProc.filterProcess(this.DDMProcesses, EVC.DDM.DDMProc.EXPECTED_STATE);
            this.Log.ExpectedStateParam(this.Log.LogCount,:) = expectedProc.getVal();
            
            
            % - log actual state parameter
            actualProc = EVC.DDM.DDMProc.filterProcess(this.DDMProcesses, EVC.DDM.DDMProc.ACTUAL_STATE);
            this.Log.ActualStateParam(this.Log.LogCount,:) = actualProc.getVal();
            
            
        end
        
    end

    
end