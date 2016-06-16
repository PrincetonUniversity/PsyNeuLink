classdef Trial < handle
    
    properties
        ID              % trial identification number
        cueID;          % cue identification of trial
        typeID;         % different types are treated as a separate trial representations in expected state space
        descr;          % description of trial
        conditions;     % array that holds a sequence of conditions that are expressed in the Trial
        params;         % holds trial specific parameters
        outcomeValues;  % array that holds values of all possible outcomes on that trial
        performance;    % array that holds all performance outcomes of the current trial
        stimSalience    % holds salience (between 0 and 1) for each stimulus 
        
        stimRespMap     % holds corresponding stimulus-response mapping
        
        probs           % to store trial outcomes
        RTs             % to store trial outcomes
        EVC             % to store trial outcomes
    end
    
    methods
        
        % constructor, arg order: type, stimSalience, outcomeValues, params
        function this = Trial(varargin)
            
            if(length(varargin) == 1)
                
                this.ID = varargin{1}.ID;
                this.cueID = varargin{1}.cueID;
                this.typeID = varargin{1}.typeID;
                this.descr = varargin{1}.descr;
                this.outcomeValues = varargin{1}.outcomeValues;
                this.stimSalience = varargin{1}.stimSalience;
                this.conditions = varargin{1}.conditions;
                this.params = varargin{1}.params;
                this.stimRespMap = varargin{1}.stimRespMap;
                if(ismember('performance',fieldnames(varargin{1})))
                   this.performance = varargin{1}.performance;
                end

                
            else
                
                % assign values
                this.ID = varargin{1};
                this.cueID = varargin{2};
                this.typeID = varargin{3};
                this.descr = varargin{4};

                if(sum(varargin{4}) ~= 1)
                    warning('Salience values should sum up to 1.0');
                end
                this.stimSalience = varargin{5};
                this.outcomeValues = varargin{6};
                if(length(varargin) >=7)
                    this.conditions = varargin{7};
                else
                    this.conditions = this.typeID;
                end
                if(length(varargin) >=8)
                    this.params = varargin{8};
                end

            end
            
        end
        
        % returns a matrix that specifies the saliency of all possible responses
        function [StimSalRespMap] = getRespSaliencyMap(this)
            
            % convert stimulus saliency into response saliency
            
            tmpStimRespMap = this.stimRespMap;
            
            [nrow ncol] = size(tmpStimRespMap);
            
            if(nrow < length(this.stimSalience))
                % assume that remaining stimuli can be ignored
                diff = length(this.stimSalience) - nrow; 
                EVC.HelperFnc.warning('Stimulus in the current trial not assigned to any response in TaskSet. Assuming that it can be ignored.');
                tmpStimRespMap = [tmpStimRespMap; zeros(diff,ncol)];
            end
            
            tmpStimRespMap = tmpStimRespMap(1:length(this.stimSalience),:);
            [~, ncol] = size(tmpStimRespMap);
            SalMap = repmat(this.stimSalience, ncol, 1);
            
            
            StimSalRespMap = times(tmpStimRespMap, transpose(SalMap));
            StimSalRespMap = StimSalRespMap/sum(StimSalRespMap(:));              % normalize
            
            
        end
        
    end
    
    % gets and sets (non-dynamic reference functions)
    methods
        
        % this.stimSalience
        function out = getStimSalience(this)
            out = this.stimSalience;
        end
        
        function setSaliency(this, newStimSalience)
            this.stimSalience = newStimSalience;
        end
        
        % this.outcomeValues
        function out = getOutcomeValues(this)
           out = this.outcomeValues; 
        end
        
        function setOutcomeValues(this, newOutcomeValues)
           this.outcomeValues =  newOutcomeValues;
        end
        
        % this.stimRespMap
        function out = getStimRespMap(this)
           out = this.stimRespMap; 
        end
        
        % this.stimRespMap
        function setStimRespMap(this, newStimRespMap)
           this.stimRespMap = newStimRespMap; 
        end
    end
    
    methods(Static)
        
        function choiceTrial = choiceTrial(IDs, cueIDs, varargin)
           
           choiceTrial.ID = IDs;
           
           if(length(varargin) >= 1) 
               choiceTrial.typeID = varargin{1};
           else
               choiceTrial.typeID = -1;
           end
           
           choiceTrial.cueID = cueIDs;

           if(length(varargin) >= 2)
               choiceTrial.descr = varargin{2};
           else
               choiceTrial.descr = 'choice';
           end

           if(length(varargin) >= 3)
               choiceTrial.conditions = varargin{3};
           else
               choiceTrial.conditions = [0];
           end           
           
           choiceTrial.outcomeValues = [];
           choiceTrial.stimSalience = [];
           choiceTrial.stimRespMap = [];
           choiceTrial.params = [];
           
           %this = EVC.Trial(choiceTrial);

        end
        
    end
    
end