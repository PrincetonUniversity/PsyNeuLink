classdef CtrlSignal < handle
    
    properties (SetAccess = public)
        Intensity             % control signal intensity 
        IntensityRange        % range of all possible control intensities for this signal
        id                    % signal identifier
        CCostFnc              % Control cost function
        adaptiveFnc           % Intensity adaption function
        useAdaptive           % flag if using an adaptive function for control intensity
        Lambda                % Reward discount factor
        EVCModel              % reference to EVCModel instance
        CtrlSigStimMap        % maps control signals (rows) to stimuli (cols)
        params                % holds specific parameters 
    end
    
    properties (Constant)
        defID = -1;             % default ID
        defRange = 0:0.1:100;   % default intensity range
    end
    
    methods
        
        % constructor: startIntensity, CtrlIntensityToCost
        function this = CtrlSignal(varargin)
            if(length(varargin) == 1)
                if(~isa(varargin{1}, 'EVC.CtrlSignal'))
                   error('A single constructor input argument has to be an instance of the class CtrlSignal for this constructor.');
                else
                    this.Intensity = varargin{1}.Intensity;
                    this.IntensityRange = varargin{1}.IntensityRange;
                    this.params = varargin{1}.params;
                    this.CCostFnc = varargin{1}.CCostFnc;
                    this.Lambda = varargin{1}.Lambda;
                    this.adaptiveFnc = varargin{1}.adaptiveFnc;
                    this.useAdaptive = varargin{1}.useAdaptive;
                    this.id = varargin{1}.id;
                    this.EVCModel = varargin{1}.EVCModel;
                    this.CtrlSigStimMap = varargin{1}.CtrlSigStimMap;
                end
            else
                if(length(varargin) >= 5)
                    if(isa(varargin{1}, 'double'))
                        this.Intensity = varargin{1}; 
                    else
                        error('Intensity needs to be a double.');
                    end
                    
                    if(isa(varargin{2}, 'EVC.EVCFnc'))
                        this.CCostFnc = varargin{2};
                    else
                        error('CCostFnc needs to be an instance of the class EVC.EVCFnc.');
                    end
                    
                    if(isa(varargin{3}, 'integer') || isa(varargin{3}, 'double'))
                        if(varargin{3} < 0 || varargin{3} > 1)
                            error('The discount factor Lamda should be between 0 and 1'); 
                        else
                            this.Lambda = varargin{3};
                        end
                    else
                        error('Lambda needs to be an integer or double variable.');
                    end
                    
                    if(isa(varargin{4}, 'integer') || isa(varargin{4}, 'double'))
                        this.CtrlSigStimMap = varargin{4};
                    else
                        error('CtrlSigStimMap needs to be an integer or double matrix.');
                    end
                    
                    if(length(varargin) >= 5)
                        if(isa(varargin{5}, 'integer') || isa(varargin{5}, 'double'))
                            this.IntensityRange = varargin{5};
                        else
                            error('intensity range needs to contain only integer or double values.');
                        end
                    else 
                        this.IntensityRange = this.defRange;
                    end
                    
                    if(length(varargin) >= 6)
                        if(isa(varargin{6}, 'integer') || isa(varargin{6}, 'double'))
                            this.id = varargin{6};
                        else
                            error('id needs to be an integer or double value.');
                        end
                    else 
                        this.id = this.defID;
                    end
                    
                else
                    
                end
                
            end
        end
        
        function setAdaptiveFnc(this, adaptiveFnc)
            
            if(isa(adaptiveFnc, 'EVCFnc') == 0)
                error('adaptiveFnc has to be an instance of the class ''EVC.Fnc''.');
            else
                this.adaptiveFnc = adaptiveFnc;
                this.useAdaptive = 1;
            end
            
        end
        
        function Intensity = getIntensity(this)
           if(this.useAdaptive)
               if(~isempty('this.adaptiveFnc', 'var'))
                   Intensity = this.adaptiveFnc(this.EVCModel, this.id) .* this.Intensity;
               else
                   error('UseAdaptive is set to 1, but no adaptiveFnc specified.');
               end
           else
               Intensity = this.Intensity; 
           end
        end
        
        function costs = getCCost(this)
            costs = this.CCostFnc.getVal(this.getIntensity());
        end
        
    end
    
    methods (Static)
        
    end
end
