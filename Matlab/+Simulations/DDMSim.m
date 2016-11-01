classdef DDMSim < Simulations.Simulation
    
    % description of class
    % runs a basic EVC DDM implementation to check general simulation functionality

    
    % global parameters
    properties
        
        % DDM simulation parameters
        defaultDDMParams                             % struct: holds all DDM params
        
        DDMProcesses                                 % struct[]: DDMProcess.type defines the DDM parameter, DDMProcess.val defines associated value (EVCFnc)
        defaultControlProcess                        % default controlled DDM process
        defaultGlobalCtrlProcess                     % default global controlled DDM process
        defaultAutomaticProcess                      % default automatic DDM process
        
        defaultControlProxy                          % default DDM parameter that control maps to
        defaultGlobalCtrlProxy                       % default DDM parameter that global control maps to
        defaultAutomaticProxy                        % default DDM parameter that automatic processing maps to
        
        defaultControlMappingFnc                     % default function that scales control intensity for a specific control parameter
        defaultGlobalCtrlMappingFnc                  % default function that scales control intensity for a global control parameter
        defaultAutomaticFnc                          % default function that maps automatic bias to DDM parameter
   
        defaultControlFnc                           % function that maps a task-specific control signal to a specific DDM parameter
        defaultGlobalCtrlFnc                        % % function that maps a global control signal to a specific DDM parameter
        
    end
    
    methods
        
        function this = DDMSim()

            import EVC.*;
            import EVC.DDM.*;
            
            % call parent constructor
            this = this@Simulations.Simulation();
            
            %% default DDM parameters

            this.defaultDDMParams.c = 0.5;                                                   % Noise coefficient
            this.defaultDDMParams.t0 = 0.45;                                                 % Non-decision time (sec)
            this.defaultDDMParams.drift = 0.3;                                                 % drift rate
            this.defaultDDMParams.bias = 0.5;                                                % (0-1; 0.5 = no bias; 1 = always produce CONTROLLED response) Assuming biased to do NON-controlled action
            this.defaultDDMParams.thresh = 1;                                                % threshold
            
            %% default DDM processes
            
            % which processes map to which DDM parameters?
            this.defaultControlProxy = DDMProc.DRIFT;                                     % default control maps to drift rate
            this.defaultGlobalCtrlProxy = DDMProc.THRESH;                                 % default global control process maps to threshold
            this.defaultAutomaticProxy = DDMProc.DRIFT;                                   % default automatic process maps to drift rate
            
            % how do model parameters (e.g. control intensity) map to DDM parameters in general?
            
            temp.defaultControlMappingFnc.type = EVCFnc.LINEAR;                                     % control-to-DDM function type
            temp.defaultControlMappingFnc.params{1} = 0;                                            % function parameter (intercept)
            temp.defaultControlMappingFnc.params{2} = 1;                                            % function parameter (slope)
            this.defaultControlMappingFnc = EVCFnc(temp.defaultControlMappingFnc.type, ...
                                            temp.defaultControlMappingFnc.params);
            
            temp.defaultGlobalCtrlMappingFnc.type = EVCFnc.LINEAR;                                  % global-control-to-DDM function type
            temp.defaultGlobalCtrlMappingFnc.params{1} = 0;                                         % function parameter (intercept)
            temp.defaultGlobalCtrlMappingFnc.params{2} = 1;                                         % function parameter (slope)
            this.defaultGlobalCtrlMappingFnc = EVCFnc(temp.defaultGlobalCtrlMappingFnc.type, ...
                                               temp.defaultGlobalCtrlMappingFnc.params);
            
            temp.defaultAutomaticFnc.type = DDMFnc.STIMBIAS;                                 % state-to-DDM function type
            temp.defaultAutomaticFnc.params{1} = [-0.5; 0.5];                                % function parameter: lower and upper bound of output DDM parameter range
            this.defaultAutomaticFnc = DDMFnc(temp.defaultAutomaticFnc.type, ...
                                               temp.defaultAutomaticFnc.params);
            
            % map all control signals to a specific DDM parameter
            temp.defaultControlFnc.type = DDMFnc.INTENSITY2DDM;
            temp.defaultControlFnc.params{1} = this.defaultCtrlSignal;
            temp.defaultControlFnc.params{2} = this.defaultControlProxy;
            temp.defaultControlFnc.params{3} = this.defaultControlMappingFnc;
            this.defaultControlFnc = DDMFnc(temp.defaultControlFnc.type, ...
                                            temp.defaultControlFnc.params);
            
            temp.defaultGlobalCtrlFnc.type = DDMFnc.INTENSITY2DDM;
            temp.defaultGlobalCtrlFnc.params{1} = this.defaultCtrlSignal;
            temp.defaultGlobalCtrlFnc.params{2} = this.defaultGlobalCtrlProxy;
            temp.defaultGlobalCtrlFnc.params{3} = this.defaultGlobalCtrlMappingFnc;                            
            this.defaultGlobalCtrlFnc = DDMFnc(temp.defaultGlobalCtrlFnc.type, ...
                                               temp.defaultGlobalCtrlFnc.params);
                                           
            % define each DDM process                            
            this.defaultControlProcess = DDMProc(DDMProc.CONTROL, ...                  % default controlled DDM process 
                                                    DDMProc.DRIFT, ...
                                                    this.defaultControlFnc);
                                           
            this.defaultGlobalCtrlProcess = DDMProc(DDMProc.CONTROL, ...               % default global controlled DDM process 
                                                       DDMProc.THRESH, ...
                                                       this.defaultGlobalCtrlFnc);                                    
                                                
            this.defaultAutomaticProcess = DDMProc(DDMProc.ACTUAL_EXPECTED, ...        % default automatic DDM process for actual & expected state
                                                    DDMProc.DRIFT, ...
                                                    this.defaultAutomaticFnc);
            
            % put all DDM processes together
            this.DDMProcesses = DDMProc.empty(2,0);
            this.DDMProcesses(1) =   this.defaultControlProcess;
            this.DDMProcesses(2) =   this.defaultAutomaticProcess;
            
            clear temp;
        end
        
        function getResults(this) 
        end
        
        function dispResults(this)
            disp('++++++++++ DDMSim ++++++++++');
            disp('[OK] Batchrun: successfull');
        end
        
        function plotTrial(this)
        end
        
        function plotSummary(this) 
        end
        
    end
    
    methods (Access = public)
        
    end
    
    methods (Access = protected)
        
        function initEVCModel(this)
            import EVC.DDM.*;
            
            this.DDMProcesses = DDMProc.addDefaultParams(this.DDMProcesses, this.defaultDDMParams);
            this.EVCM = EVCDDM(this.ctrlSignals, this.taskEnv, this.DDMProcesses);
        end
        
        function initCtrlSignals(this)
           this.ctrlSignals(1) = this.defaultCtrlSignal; 
        end
        
    end
    
end

