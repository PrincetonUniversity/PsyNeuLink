classdef CtrlSigAdapt < EVC.EVCFnc
    
    % This class implements adaptivity functions that may simulate adaption effects of control identities 
    % The domain for each specific function may represent adaption steps (e.g. trials)
    % The codomain for each specific function should also not exceed the interval (0,1).
    
    
    % HOW TO ADD A COST FNC:
    % 1) add cost fnc @EVCFnc.m
    
    properties (SetAccess = private)
        trace;                          % integer value > 0: how many trials back in the past shall be taken in to consideration?; if 0: take whole history
        adaptionPerTrial;               % how many "steps" to adapt for each trial a control identity has been applied
    end
    
    properties (Constant)
        startTrace = 1;
        startAdaptionPerTrial = 1;
    end
    
    methods
        
        %constructor: type refers to the type of function (linear, exponential), params are parameters
        % varargin order: trace, adaptionPerTrial recoveryPerTrial, reset
        function this = CtrlSigAdapt(type, params, varargin)
            
            
          % call superclass constructor to create an EVC function instance
          this = this@EVC.EVCFnc(type, params);
          
          % use arguments if passed, else use default constants
          
          if(~isempty(varargin))
            this.trace = varargin{1};
          else
            this.trace = this.startTrace;
          end
          
          if(length(varargin)>=2)
            this.adaptionPerTrial = varargin{2};
          else
            this.adaptionPerTrial = this.startAdaptionPerTrial;
          end
          
        end
        
    end
    
    methods (Access = public)
        
        % calculates adaption dependend on cost function used
        % param order: EVCModel || TaskEnv, CurrSigIdx
        function out = getAdaptedOutput(this, varargin)
            
            if(length(varargin) >= 2) % treat argument as EVC.EVCModel instance
                
               if(~isa(varargin{1}, 'EVC.EVCModel') || ~isa(varargin{2}, 'double')) error('getAdaptedCosts requires either an EVCModel instance and a corresponding control signal index as arguments.'); end
               Log = varargin{1}.Log;
               CurrSigIdx = varargin{2};
               
            else
               error('getAdaptedCosts requires either an EVCModel instance and a corresponding control signal index as arguments.'); 
            end
            
            % extract control signal indices from signal log
            for i = 1:size(Log.CtrlSignals,1)
                SigCol = find(Log.SignalIdxs(i,:) == CurrSigIdx);
                
                if(~isempty(SigCol))
                    if(Log.CtrlIntensities(i,SigCol) == max(Log.CtrlIntensities(i,:)))
                        IntensitySeq(i,:) = Log.CtrlIntensities(i,SigCol);
                    else
                        IntensitySeq(i,:) = 0;
                    end
                else
                    IntensitySeq(i,:) = 0;
                end
            end
            
            
            
            if(~isempty(Log.CtrlSignals))
            
                % if this.trace == 0, use whole sequence
                actualTrace = EVC.HelperFnc.ifelse(this.trace == 0, size(IntensitySeq,1), this.trace);
                actualTrace = min(actualTrace, size(IntensitySeq,1));
                % calculate x steps on adaption function
                % for now: order of adaption and recovery trials doesn't matter -> TODO: discuss discount factors

                adaption = this.adaptionPerTrial * sum(IntensitySeq((size(IntensitySeq,1)+1-actualTrace):size(IntensitySeq,1),:));
                steps = max(0, adaption);
                
                out = this.getOutput(steps);
                                
                disp('adaption stuff');
                disp(IntensitySeq);
                disp(size(IntensitySeq,1));
                disp(sum(IntensitySeq((size(IntensitySeq,1)+1-actualTrace):size(IntensitySeq,1),:)));
                disp(out);
                disp('---');
            
            else
               out = 1; % no adaptation 
            end
            
        end
       
    end
    
end