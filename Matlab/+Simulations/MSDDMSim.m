classdef MSDDMSim < Simulations.Simulation
    
    % description of class
    % runs a basic EVC DDM implementation to check general simulation functionality

    
    % global parameters
    properties
        
        % DDM simulation parameters
        defaultDDMParams                             % struct: holds all DDM params
        
        MSDDMProcesses                                 % struct[]: MSDDMProcess.type defines the DDM parameter, MSDDMProcess.val defines associated value (EVCFnc)
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
        
        defaultControlDuration                      % stage duration for default control process
        defaultGlobalCtrlDuration                   % stage duration for global control process
        defaultAutomaticDuration                    % stage duration for automatic process
        
        defaultControlStage                         % stage at which task-related control kicks in
        defaultGlobalCtrlStage                      % stage at which global control kicks in
        defaultAutomaticStage                       % stage at which automatic bias kicks in
    end
    
    methods
        
        function this = MSDDMSim()

            import EVC.*;
            import EVC.MSDDM.*;
            
            % call parent constructor
            this = this@Simulations.Simulation();
            
            %% default DDM parameters

            this.defaultDDMParams.c = 0.5;                                                   % Noise coefficient
            this.defaultDDMParams.t0 = 0.200;                                                 % Non-decision time (sec)
            this.defaultDDMParams.drift = 0.3;                                                 % drift rate
            this.defaultDDMParams.bias = 0;                                                % (-1 ... 1; 0 = no bias; 1 = always produce CONTROLLED response) Assuming biased to do NON-controlled action
            this.defaultDDMParams.thresh = 1;                                                % threshold
            
            this.defaultDDMParams.durations = [0.2 0];                                     % durations for each stage
            
            %% default DDM processes
            
            % define MSDDM durations for 2 stages
            % note that the duration for each stage can be associated with
            % any dynamic EVC value
            temp.defaultDuration(1) = EVC.EVCFnc(EVC.EVCFnc.VALUE, {this.defaultDDMParams.durations(1)});     % for automatic processing
            temp.defaultDuration(2) = EVC.EVCFnc(EVC.EVCFnc.VALUE, {this.defaultDDMParams.durations(2)});     % for controlled processing
            
            
            % which processes map to which DDM parameters?
            this.defaultControlProxy = MSDDMProc.DRIFT;                                     % default control maps to drift rate
            this.defaultGlobalCtrlProxy = MSDDMProc.THRESH;                                 % default global control process maps to threshold
            this.defaultAutomaticProxy = MSDDMProc.DRIFT;                                   % default automatic process maps to drift rate
            
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
            
            temp.defaultAutomaticFnc.type = MSDDMFnc.STIMBIAS;                                 % state-to-DDM function type
            temp.defaultAutomaticFnc.params{1} = [-0.5; 0.5];                                % function parameter: lower and upper bound of output DDM parameter range
            this.defaultAutomaticFnc = MSDDMFnc(temp.defaultAutomaticFnc.type, ...
                                               temp.defaultAutomaticFnc.params);
            this.defaultAutomaticStage = 1;                                                 % automatic processing during 1st stage                               
            this.defaultAutomaticDuration = temp.defaultDuration(1);
            
            % map all control signals to a specific DDM parameter
            temp.defaultControlFnc.type = MSDDMFnc.INTENSITY2DDM;
            temp.defaultControlFnc.params{1} = this.defaultCtrlSignal;
            temp.defaultControlFnc.params{2} = this.defaultControlProxy;
            temp.defaultControlFnc.params{3} = this.defaultControlMappingFnc;
            this.defaultControlFnc = MSDDMFnc(temp.defaultControlFnc.type, ...
                                            temp.defaultControlFnc.params);
            this.defaultControlStage = 2;                                                   % control kicks in the 2nd stage
            this.defaultControlDuration = temp.defaultDuration(2);
            
            temp.defaultGlobalCtrlFnc.type = MSDDMFnc.INTENSITY2DDM;
            temp.defaultGlobalCtrlFnc.params{1} = this.defaultCtrlSignal;
            temp.defaultGlobalCtrlFnc.params{2} = this.defaultGlobalCtrlProxy;
            temp.defaultGlobalCtrlFnc.params{3} = this.defaultGlobalCtrlMappingFnc;
            this.defaultGlobalCtrlFnc = MSDDMFnc(temp.defaultGlobalCtrlFnc.type, ...
                                               temp.defaultGlobalCtrlFnc.params);
                                           
            this.defaultGlobalCtrlStage = [1 2];                                                % global control acts on all stages
            this.defaultGlobalCtrlDuration = temp.defaultDuration(2);
            
            % define each DDM process                            
            this.defaultControlProcess = MSDDMProc(MSDDMProc.CONTROL, ...                  % default controlled DDM process 
                                                    MSDDMProc.DRIFT, ...
                                                    this.defaultControlFnc, ...
                                                    this.defaultControlStage, ...
                                                    this.defaultControlDuration);
                                           
            this.defaultGlobalCtrlProcess = MSDDMProc(MSDDMProc.CONTROL, ...               % default global controlled DDM process 
                                                       MSDDMProc.THRESH, ...
                                                       this.defaultGlobalCtrlFnc, ...
                                                       this.defaultGlobalCtrlStage, ...
                                                       this.defaultGlobalCtrlDuration);                                    
                                                
            this.defaultAutomaticProcess = MSDDMProc(MSDDMProc.ACTUAL_EXPECTED, ...        % default automatic DDM process for actual & expected state
                                                    MSDDMProc.DRIFT, ...
                                                    this.defaultAutomaticFnc, ...
                                                    this.defaultAutomaticStage, ...
                                                    this.defaultAutomaticDuration);
            
            % put all DDM processes together
            this.MSDDMProcesses = MSDDMProc.empty(2,0);
            this.MSDDMProcesses(1) =   this.defaultControlProcess;
            this.MSDDMProcesses(2) =   this.defaultAutomaticProcess;
            
            clear temp;
        end
        
        function getResults(this) 
        end
        
        function dispResults(this)
            disp('++++++++++ MSDDMSim ++++++++++');
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
            import EVC.MSDDM.*;
            
            this.MSDDMProcesses = MSDDMProc.addDefaultParams(this.MSDDMProcesses, this.defaultDDMParams);
            this.EVCM = EVCMSDDM(this.ctrlSignals, this.taskEnv, this.MSDDMProcesses);
        end
        
        function initCtrlSignals(this)
           this.ctrlSignals(1) = this.defaultCtrlSignal; 
        end
        
    end
    
end

