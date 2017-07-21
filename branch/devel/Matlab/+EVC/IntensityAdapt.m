classdef IntensityAdapt < EVC.EVCFnc
    
    % This class implements adaptivity functions that may simulate adaption effects that depend on the amount of control intensity applied 
    % The domain for each specific function may represent adaption steps (sum of all applied intensity values)
    % The codomain for each specific function should also not exceed the
    % interval (0,1)
    
    
    % HOW TO ADD A COST FNC:
    % 1) add cost fnc @EVCFnc.m
    
    properties (SetAccess = private)
        trace;                          % integer value > 0: how many trials back in the past shall be taken in to consideration?; if 0: take whole history
        adaptionScale;                  % scaling parameter for sum of applied control intensities
    end
    
    properties (Constant)
        startTrace = 1;
        startAdaptionScale = 1;
    end
    
    methods
        
        %constructor: type refers to the type of function (linear, exponential), params are parameters
        % varargin order: trace, adaptionPerTrial recoveryPerTrial, reset
        function this = IntensityAdapt(type, params, varargin)
            
            
          % call superclass constructor to create an EVC function instance
          this = this@EVC.EVCFnc(type, params);
          
          % use arguments if passed, else use default constants
          
          if(~isempty(varargin))
            this.trace = varargin{1};
          else
            this.trace = this.startTrace;
          end
          
          if(length(varargin)>=2)
            this.adaptionScale = varargin{2};
          else
            this.adaptionScale = this.startAdaptionScale;
          end
          
        end
        
    end
    
    methods (Access = public)
        
        % calculates adaption depending on sum of applied control
        % intensities
        % param order: EVCSim
        function out = getAdaptedOutput(this, varargin)
            
            if(length(varargin) == 1) % treat argument as EVC.EVCSim instance
                
               if(~isa(varargin{1}, 'EVC.EVCSim')) error('getAdaptedCosts requires either an EVCSim instance as argument or both a EVC.TaskEnv instance and a corresponding control signal index.'); end    
               Log = varargin{1}.Log;
               
            else
                
               error('There is no implementation of this function for more than 1 input parameter yet.');
               
            end
            
            
            if(~isempty(Log.CtrlIntensities))
            
                % if this.trace == 0, use whole sequence
                actualTrace = EVC.HelperFnc.ifelse(this.trace == 0, length(Log.CtrlIntensities), this.trace);

                % calculate actual trace based on past sequence

                actualTrace = min(length(Log.CtrlIntensities), actualTrace);

                % calculate x steps on adaption function
                % for now: order of adaption and recovery trials doesn't matter -> TODO: discuss discount factors

                adaption = this.adaptionScale * sum(Log.CtrlIntensities((length(Log.CtrlIntensities)+1-actualTrace):length(Log.CtrlIntensities)));


                out = this.getOutput(adaption);
            
            else
               out = this.getOutput(0); 
            end
            
        end
       
    end
    
end