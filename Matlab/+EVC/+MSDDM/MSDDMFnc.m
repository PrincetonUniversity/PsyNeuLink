classdef MSDDMFnc < EVC.EVCFnc
    
    % this class implements DDM-specific functions
    % most of the functions may reference an EVCDDM instance

    
    properties (Constant)
        STIMBIAS = 1;           % map saliency to range
        INTENSITY2DDM = 2;
        PREV_INTENSITY2DDM = 3;
        
        % DDM poxies
        DRIFT = EVC.DDM.DDMProc.DRIFT;
        THRESH = 2;
        BIAS = 3;
        NOISE = 4;
        T0 = 5;
        
        % holds amount of required parameters for each function type
        paramReqDDM = [1 ...   STIMBIAS: 1) lower and upper bound of DDM parameter range, i.e. [min max]
                       3 ...
                       2];         
    end
    
    methods
        
        function this = MSDDMFnc(type, params, varargin)
            
          % call superclass constructor to create an EVC function instance
          this = this@EVC.EVCFnc(type, params);
          
          [inputVal EVCM] = this.extractInput(varargin);
          this.input = inputVal;
          this.EVCModel = EVCM;
        end
        
    end
    
    methods (Access = public)
        
        % calculates output value dependent on specified function type
        function out = getVal(this, varargin)
            
            [inputVal, EVCM] = this.extractInput(varargin);
            
            if(this.type == this.STIMBIAS)
               out = getStimBias(this, EVCM); 
            end
            
            if(this.type == this.INTENSITY2DDM)
               out =  intensityToDDM(this, EVCM);
            end
            
            if(this.type == this.PREV_INTENSITY2DDM)
               out =  prevIntensityToDDM(this, EVCM);
            end
            
        end
       
    end
    
    methods (Access = private)
        
        %% function definitions
        
        % STIMBIAS... returns a response bias for the specified interval
        % param{1}: [b1 b2] (lower and upper bound of output DDM parameter range)
        function out = getStimBias(this, EVCM)
            
            % determine which state to operate on
            if(EVCM.useActualState)
                currTrial = EVCM.State.Actual;
            else
                currTrial = EVCM.State.Expected;
            end

            StimSalRespMap = currTrial.getRespSaliencyMap();
            
            salR1 = sum(StimSalRespMap(:,1));
            salR2 = sum(sum((StimSalRespMap(:,2:length(StimSalRespMap(1,:)))))); %FIX THIS: all other responses than R1 may belong to the lower threshold
            
            rangeWidth = this.params{1}(2) - this.params{1}(1);

            % how much bias toward upper threshold?
            R1Delta = + rangeWidth/2*salR1;
            
            % how much bias toward lower threshold?
            R2Delta = -rangeWidth/2*salR2;
            
            middleOfRange = this.params{1}(1) + (rangeWidth)/2;

            %EVC.HelperFnc.disp(source, 'middleOfRange', middleOfRange, 'rangeWidth', rangeWidth, 'R1Delta', R1Delta, 'R2Delta', R2Delta); % DIAGNOSIS 
            
            out = middleOfRange + R1Delta + R2Delta;
           
        end
        
        % PREV_INTENSITY2DDM... maps control signal intensity to DDM parameter
        % param{1}: integer (refers to DDM proxy, see constants)
        % param{2}: EVC.EVCFnc (mapping function)
        function out = prevIntensityToDDM(this, EVCM)
            
            prevIntensities = EVCM.System.previousIntensities;
            numSignals = size(prevIntensities,1);
            
            impact = zeros(1, numSignals);
            
            for i = 1:numSignals
                
                currSignal = EVCM.System.CtrlSignals(i);
            
                % mapping function
                if(~isa(this.params{2}, 'EVC.EVCFnc'))
                   error('Second EVCFnc parameter has to be an instance of EVC.EVCFnc (intensity to DDM mapping function)');
                else
                    mappingFnc = this.params{2};
                end

                % get signal 

                % for drift & bias parameters, calculate in which direction control acts
                % (+1 -> upper threshold, -1 -> lower threshold)
                if(this.params{2} == this.DRIFT || this.params{2} == this.BIAS)

                    % how many stimuli are available?
                    ctrlSigStimMap = currSignal.CtrlSigStimMap;
                    stimRespMap = EVCM.currentTaskSet();
                    [nStimuliEnv,~] = size(stimRespMap);
                    [nStimuliCtrl] = length(ctrlSigStimMap);

                    % check if maps correspond in size
                    if(nStimuliCtrl < nStimuliEnv)
                        ctrlSigStimMap = [ctrlSigStimMap repmat(0,1,nStimuliEnv-nStimuliCtrl) ];
                        warning('CtrlSigStimMap and stimRespMap don''t match. Filling missing parts with zero.');
                    end

                    % 
                    if(nStimuliCtrl > nStimuliEnv)
                        ctrlSigStimMap = ctrlSigStimMap(1:nStimuliEnv);
                        warning('CtrlSigStimMap and stimRespMap don''t match. Cutting off excess of Stimuli.');
                    end

                    % translate both maps into control-to-response map
                    CtrlIToResp = transpose(stimRespMap) * transpose(ctrlSigStimMap);

                    % make sure that there are only 2 possible responses (DDM default)
                    if(length(CtrlIToResp) < 2)
                        CtrlIToResp = [1 0]; % by default the current control intensity acts in favor of the upper response
                    end

                    % translate control-to-response map into scalar value
                    % (positive: upper response; negative: bottom response)
                    CtrlDirection = CtrlIToResp(1) - sum(CtrlIToResp(2:length(CtrlIToResp)));           
                else
                   CtrlDirection = 1;
                end

                impact(i) = CtrlDirection * mappingFnc.getVal(prevIntensities(i,:));

            end
            
            out = sum(impact);
        end
        
        % INTENSITY2DDM... maps control signal intensity to DDM parameter
        % param{1}: EVC.CtrlSignal
        % param{2}: integer (refers to DDM proxy, see constants)
        % param{3}: EVC.EVCFnc (mapping function)
        function out = intensityToDDM(this, EVCM)
            
            % current control signal
            if(~isa(this.params{1}, 'EVC.CtrlSignal'))
               error('First EVCFnc parameter has to be an instance of EVC.CtrlSignal (control signal instnace with corresponding intensity)');
            else
                currSignal = this.params{1};
            end
            
            % mapping function
            if(~isa(this.params{3}, 'EVC.EVCFnc'))
               error('Second EVCFnc parameter has to be an instance of EVC.EVCFnc (intensity to DDM mapping function)');
            else
                mappingFnc = this.params{3};
            end
            
            % for drift & bias parameters, calculate in which direction control acts
            % (+1 -> upper threshold, -1 -> lower threshold)
            if(this.params{2} == this.DRIFT || this.params{2} == this.BIAS)
            
                % how many stimuli are available?
                ctrlSigStimMap = currSignal.CtrlSigStimMap;
                stimRespMap = EVCM.currentTaskSet();
                [nStimuliEnv,~] = size(stimRespMap);
                [nStimuliCtrl] = length(ctrlSigStimMap);
                
                % check if maps correspond in size
                if(nStimuliCtrl < nStimuliEnv)
                    ctrlSigStimMap = [ctrlSigStimMap repmat(0,1,nStimuliEnv-nStimuliCtrl) ];
                    warning('CtrlSigStimMap and stimRespMap don''t match. Filling missing parts with zero.');
                end
            
                % 
                if(nStimuliCtrl > nStimuliEnv)
                    ctrlSigStimMap = ctrlSigStimMap(1:nStimuliEnv);
                    warning('CtrlSigStimMap and stimRespMap don''t match. Cutting off excess of Stimuli.');
                end
                
                % translate both maps into control-to-response map
                CtrlIToResp = transpose(stimRespMap) * transpose(ctrlSigStimMap);
                
                % make sure that there are only 2 possible responses (DDM default)
                if(length(CtrlIToResp) < 2)
                    CtrlIToResp = [1 0]; % by default the current control intensity acts in favor of the upper response
                end
                
                % translate control-to-response map into scalar value
                % (positive: upper response; negative: bottom response)
                CtrlDirection = CtrlIToResp(1) - sum(CtrlIToResp(2:length(CtrlIToResp)));           
            else
               CtrlDirection = 1;
            end
            
            out = CtrlDirection * mappingFnc.getVal(currSignal.getIntensity());
        end
    end
    
end