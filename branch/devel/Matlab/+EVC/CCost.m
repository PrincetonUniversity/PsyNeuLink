classdef CCost < EVC.EVCFnc
    
    % This class implements Cost functions that transform control intensity values into a cost parameter. 
    % The domain for each specific function should not exceed the interval (0,1).
    % The codomain for each specific function should also not exceed the interval (0,1).
    
    
    % HOW TO ADD A COST FNC:
    % 1) add cost fnc @EVCFnc.m
    
    properties
        useLimit                % flag if using limited costs (between 0 and 1)
        useAdaptive             % flag if using an adaptive cost function for control signal
        adaptiveFnc             % adaptive func
    end
    
    properties (Constant)
        startAdaptive = 0;
        startLimit = 0;
    end
    
    methods
        
        %constructor: type refers to the type of function (linear, exponential), params are parameters
        function this = CCost(type, params)
            
          % call superclass constructor to create an EVC function instance
          this = this@EVC.EVCFnc(type, params);
          
          this.useAdaptive = this.startAdaptive;
          this.useLimit = this.startLimit;
        end
        
        % sets up an adaptive function
        function setAdaptiveFnc(this, adaptiveFnc)
            
            if(isa(adaptiveFnc, 'EVC.CtrlSigAdapt') == 0)
                error('adaptiveFnc has to be an instance of the class ''EVC.CtrlSigAdapt''.');
            else
                this.adaptiveFnc = adaptiveFnc;
                this.useAdaptive = 1;
            end
            
        end
        
    end
    
    methods (Access = public)
        
        % calculates control cost dependend on cost function used
        function cost = getCCost(this, CtrlSignal)
            
            if(isa(CtrlSignal, 'EVC.CtrlSignal') == 0)
                error('CtrlSignal has to be an instance of the class ''EVC.CtrlSignal''.');
            end

            % use specified superclass function
            cost = getVal(this, CtrlSignal.getIntensity());
            
            % limit costs
            if(this.useLimit)
                cost = this.limitCosts(cost);
            end
            
            % if using adaptive cost fnc
            if(this.useAdaptive)
                if(~isa(this.adaptiveFnc, 'EVC.CtrlSigAdapt'))
                    error('Error using adaptive cost caluclation: Please set up adaptive Fnc using CCost.setAdaptiveFnc()');
                end
                if(~isa(CtrlSignal.EVCmodel, 'EVC.EVCSim'))
                    error('Error using adaptive cost caluclation: Please set up EVCSim reference for corresponding control signal.');
                end
                if(~isnumeric(CtrlSignal.id))
                    error('Error using adaptive cost caluclation: Please set up ID for corresponding control signal.');
                end
                
                cost = this.adaptiveFnc.getAdaptedOutput(CtrlSignal.EVCmodel, CtrlSignal.id) .* cost;
            end

        end
       
    end
    
    methods (Static = true)
       
        function limitedCosts = limitCosts(cost)
           limitedCosts = cost;
           limitedCosts(limitedCosts > 1) = 1;
           limitedCosts(limitedCosts < 0) = 0;
        end
        
    end
end