classdef EVCFnc < handle
    
    % This class implements control functions that may be used as control cost functions or adaptive functions 
    
    
    % HOW TO ADD A CONTROL FNC:
    % 1) add constant term to constant properties section
    % 2) add required amount of parameters to paramReq
    % 2) add actual function to methods section
    % 3) add call for new function to getVal method
    
    properties (SetAccess = public)
        type                    % defines the type of cost function (linear/exponential/...) using the constant properties below
        params                  % cell array of parameters needed to describe the function
        input                   % fnc handle{} -> points to dynamic input value
        EVCModel                % EVCModel: instance of an EVC model the function may refer to
        actualParamReq          % holds amount of required parameters for each function type
    end
    
    properties (Constant)
        VALUE = 1;
        LINEAR = 2;
        EXP = 3;
        POW = 4;
        REWRATE = 5;
        FORAGEVAL = 6;
        
        % holds amount of required parameters for each function type
        paramReq = [1, ... VALUE:       f(x) = p1
                    2, ... LINEAR:      f(x) = p1 + p2 * x
                    2, ... EXP:         TODO: restructure 
                    3, ... POW:         f(x) = p1 * (x + p2)^p3
                    2, ... REWRATE:     f(x) = sum(expectedValue(outcome) / (p1 + p2 * reactionTime(outcome)), outcomes)
                    2  ... FORAGEVAL:   do be defined
                    ];   
    end
    
    methods
        
        %constructor: type refers to the type of function (linear, exponential, etc.), params are parameters
        function this = EVCFnc(type, params, varargin)
            
            this.actualParamReq = this.paramReq;
            this.type = type;
            this.params = params; 
            this.EVCModel = NaN;
            this.input = {};
            
            for i = 1:length(varargin)
                if(isa(varargin{i}, 'EVC.EVCModel'))
                    this.EVCModel = varargin{i};
                end
                if(isa(varargin{i}, 'function_handle'))
                    this.input{i} = varargin{i};
                end
                if(isa(varargin{i}, 'cell'))
                    this.input = varargin{i};
                end
                if(isa(varargin{i}, 'char'))
                    if(strcmp(varargin{i}, 'paramReq') && length(varargin) >= i+1)
                        this.actualParamReq = eval(strcat('this.',varargin{i+1}));
                    end
                end
            end
            
            if(isempty(this.actualParamReq))
                this.actualParamReq = this.paramReq;
            end
            
            % check if there are enough params with regard to the function
            % type
            if(length(params) < this.actualParamReq(type))
               error('The params array doesn''t contain enough parameters for the specified function type');
            end
            
        end
        
    end
    
    methods (Access = public)
        
        % calculates output value dependent on specified function type
        function out = getVal(this, varargin)
            
            [inputVal, EVCM] = this.extractInput(varargin);
            
            if(this.type == this.VALUE)
               out = getValOut(this, inputVal); 
            end

            if(this.type == this.LINEAR)
                out = getLinearOut(this, inputVal);
            end

            if(this.type == this.EXP)
                out = getExpOut(this, inputVal);
            end

            if(this.type == this.POW)
                out = getPowOut(this, inputVal);
            end
            
            if(this.type == this.REWRATE)
                out = getRewardRate(this, inputVal);
            end
            
            if(this.type == this.FORAGEVAL)
                out = getForagingValue(this, inputVal);
            end
            
        end
       
        
        % specific control functions
        
        % decompose input parameters
        function [inputVal EVCM] = extractInput(this, input)
            
            inputVal = [];
            EVCM = [];
            
            for i = 1:length(input)
                if(isa(input{i}, 'EVC.EVCModel'))
                    EVCM = input{i};
                else
                % if(isnumeric(input{i}))      % don't restrict input to numeric input
                    inputVal = input{i};
                end
            end
            
           % if no input provided, use dynamic input from fnc handle
           if(isempty(inputVal))
              for i = 1:length(this.input)
                    inputVal(i) = this.input{i}(); 
              end
           end
           
           if(isempty(EVCM))
              EVCM = this.EVCModel; 
           end
            
        end
        
    end
    
    methods (Access = private)
        
        
        
        %% function definitions
        
        % VAL: return value
        function out = getValOut(this, inputVal)
            
           if(isempty(inputVal))
               out = this.params{1};
           else
               out = inputVal;
           end
           
        end
        
        % LINEAR: linear function
        function out = getLinearOut(this, inputVal)
            
           out =   this.params{1} + inputVal .* this.params{2};
           
        end
        
        % EXP: exponential function
%         function out = getExpOut(this, inputVal)
%             
%            x = this.params(1)+ (this.params(2)-this.params(1)).*inputVal;
%            out = (exp(x)-1)./exp(this.params(2));
%            
%         end
        function out = getExpOut(this, inputVal)
            
            out = exp(this.params{1}*inputVal+this.params{2});
            
        end
        
        % POW: power function
        function out = getPowOut(this, EVCM)
            
            out = this.params{1} .* ((inputVal + this.params{2}).^(this.params{3}));
            
        end
        
        % REWRATE: Reward Rate ... calculate expected value of an outcome
        % divided by weighted reaction time (+ offset) for this outcome;
        % sum values over all outcomes
        % inputVal ...current state
        function out = getRewardRate(this, inputVal)
            % trial perfromance
            performance = inputVal;
            
            % pre-specified trial outcomes
            if(this.EVCModel.useActualState)
                values = this.EVCModel.State.Actual.outcomeValues;
            else
                values = this.EVCModel.State.Expected.outcomeValues;
            end
            
            % sum up for each control signal outcome:
            % Probability(outcome|CtrlSignal, State) * Value(outcome)
            expectedValue  = sum(performance.probs .* repmat(transpose(values),1,size(performance.probs,2)),1);
            
            % normalize Reward by weighted average RT
            meanRTs = mean(performance.probs.*performance.RTs,1);
            expectedValue = expectedValue./(this.params{1} + meanRTs*this.params{2});
            
            out = expectedValue;
        end
        
        function out = getForagingValue(this, inputVal)
           
            performance = inputVal;
            
            % choiceProbabilities: 
            % 1st value indicates probability of hitting the upper threshold (choosing to harvest)
            % 2nd value indicates probability of hitting the lower threshold (choosing to switch)
            choiceProbabilities = performance.probs;
            
            % choiceRTs:
            % 1st value is RT associated with harvesting
            % 2nd value is RT associated with switching (travel time NOT included!)
            if(this.EVCModel.State.TaskEnv.CurrTrialIdx > 1)
                numTrials = this.EVCModel.State.TaskEnv.CurrTrialIdx-1;
                harvestRTs = this.EVCModel.Log.RTs(1:numTrials,1);
                switchRTs = this.EVCModel.Log.RTs(1:numTrials,2);
                harvestChoices = ~logical(this.EVCModel.Log.ERs(1:numTrials,2));
                rewards = [this.EVCModel.Log.Trials(1:numTrials).outcomeValues];
                rewards = reshape(rewards, 2, numTrials);
                rewards = rewards(1,:);
                avgRewardRate = sum(rewards(harvestChoices))/ ...
                        (sum(harvestRTs(harvestChoices)) + 7.43*sum(harvestChoices) + sum(switchRTs(~harvestChoices))  +  sum(~harvestChoices) * 10);
            
            else
                avgRewardRate = 0;
            end
                
            choiceRTs = performance.RTs;
            % this might be a good place to add the travel time to 2nd option
            
            % calculate value for corresponding option here
            harvestValue = this.EVCModel.State.Expected.outcomeValues(1)./(mean(choiceRTs) +.43 + 7);
            foragingValue = repmat(avgRewardRate, size(harvestValue));
            if(length(harvestValue) == 1)
                %disp(['harvest value is' num2str(harvestValue) ' and foraging value is' num2str(foragingValue)]);
            end
            
            out = choiceProbabilities(1) * harvestValue + choiceProbabilities(2) * foragingValue;
            
        end
        
    end
    
end