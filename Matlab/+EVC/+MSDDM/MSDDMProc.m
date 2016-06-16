classdef MSDDMProc < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        type                % ENUM: which type of process (control/actual state/expected state)?
        DDMProxy            % ENUM: to which DDM parameter does the process map?
        input               % EVCFnc: input to DDM parameter 
        stage               % double[]: indicates DDM stage
        duration            % EVCFnc: indicates duration in seconds of corresponding DDM stage
    end
    
    % enumerators 
    properties (Constant)
        % process types
        DEFAULT = 0;
        CONTROL = 1;
        ACTUAL_STATE = 2;
        EXPECTED_STATE = 3;
        ACTUAL_EXPECTED = [2 3];
        
        % DDM poxies
        DRIFT = 1;
        THRESH = 2;
        BIAS = 3;
        NOISE = 4;
        T0 = 5;
    end
    
    methods
        
        function this = MSDDMProc(type, DDMProxy, input, stage, varargin)
            
           % check inputs
           if(isnumeric(type))
               for i = 1:length(type)
                    if(type(i) < 0 || type(i) > 3)
                        error('There is no process constant with the specified process type.');
                    else
                        this.type(i) = type(i); 
                    end
               end
           else
               error('First input parameter (process type) needs to be a numeric value');
           end
           
           if(isnumeric(DDMProxy))
               for i = length(DDMProxy)
                   if(DDMProxy(i) < 1 || DDMProxy(i) > 5)
                    error('There is no process constant with the specified process type.');
                   else
                       this.DDMProxy(i) = DDMProxy(i);
                   end
               end
           else
               error('Second input parameter (DDM proxy) needs to be a numeric value');
           end
           
           if(~isa(input, 'EVC.EVCFnc'))
               error('Third input parameter needs to be an instance of class EVC.EVCFnc');
           else
              this.input = input; 
           end
           
           this.stage = stage;
           
           % specify duration
           if(~isempty(varargin))
              if(isa(varargin{1}, 'EVC.EVCFnc'))
                  this.duration = varargin{1};
              else
                  error('Additional parameter may dynamically indicate the stage duration in seconds as an instance of EVC.EVCFnc.');
              end
           end
            
        end
        
        % returns outcome value
        function out = getVal(this)
           out = this.input.getVal(); 
        end
        
        % returns duration value
        function out = getDuration(this)
            out = this.duration.getVal();
        end
        
    end
    
    methods (Static = true)
        
        % return processes with specified process types
        % specify filter criterion 'type', 'DDMProxy' or 'stage' followed
        % by corresponding array of allowed values (multiple filter specifications
        % possible)
        function out = filterProcess(processes, varargin)
            
            import EVC.MSDDM.*;
            
            if(~isa(processes, 'EVC.MSDDM.MSDDMProc'))
               error('First input argument needs to be an instance of EVC.MSDDM.MSDDMProcess.'); 
            end
            
            allowedTypes = [MSDDMProc.DEFAULT MSDDMProc.CONTROL MSDDMProc.ACTUAL_STATE MSDDMProc.EXPECTED_STATE];
            allowedDDMProxies = [MSDDMProc.DRIFT MSDDMProc.THRESH MSDDMProc.BIAS MSDDMProc.NOISE MSDDMProc.T0];
            allowedStages = NaN;
            
            currFilter = '';
            for i = 1:length(varargin)
                if(ischar(varargin{i}))
                    currFilter = varargin{i};
                end
                if(isnumeric(varargin{i}) && ~isempty(currFilter))
                    switch currFilter
                        case 'type'
                           allowedTypes = varargin{i};
                        case 'DDMProxy'
                           allowedDDMProxies = varargin{i};
                        case 'stage'
                            allowedStages = varargin{i};
                    end
                end
                
            end

            out = EVC.MSDDM.MSDDMProc.empty(1,0);
            outIndex = 1;
            
            % only add processes with allowed process types
            for i = 1:length(processes)
                % true if at least one of the process types is a member of allowed types
                if(ismember(true,ismember(processes(i).type, allowedTypes)) && ...
                        ismember(processes(i).DDMProxy, allowedDDMProxies))
                   if(ismember(true,ismember(processes(i).stage, allowedStages)) || isnan(allowedStages))
                        out(outIndex) = processes(i);
                        outIndex = outIndex + 1;
                   end
                end
            end
            
        end
        
        function stages = getStages(processes)
           
            stages = [];
            for i = 1:length(processes)
               % add stage if not already found
               addIdx = ~ismember(processes(i).stage, stages);
               stages = [stages processes(i).stage(addIdx)];
            end
            
            % order stages
            stages = sort(stages);
            
        end
        
        % add default processes for not-specified DDM processes
        function out = addDefaultParams(processes, defaultParams)
            
            import EVC.MSDDM.*;
            
            if(~isa(processes, 'EVC.MSDDM.MSDDMProc'))
               error('First input argument needs to be an instance of EVC.MSDDM.MSDDMProcess.'); 
            end
            
            % get number of stages
            stages = MSDDMProc.getStages(processes);
            
            % for each stage, specify missing parameters
            for curr_stageIdx = 1:length(stages)
                
                curr_stage = stages(curr_stageIdx);
                curr_processes = MSDDMProc.filterProcess(processes, 'stage', curr_stage);
                
                % which DDM parameters aren't specified yet?
                if(curr_stageIdx == 1)
                    % first stage needs to contain starting point & T0
                    missingDDMProxies = [EVC.DDM.DDMProc.DRIFT EVC.DDM.DDMProc.THRESH EVC.DDM.DDMProc.BIAS EVC.DDM.DDMProc.NOISE EVC.DDM.DDMProc.T0];
                else
                    missingDDMProxies = [EVC.DDM.DDMProc.DRIFT EVC.DDM.DDMProc.THRESH EVC.DDM.DDMProc.NOISE];
                end
                
                for i = 1:length(curr_processes)
                    missingDDMProxies = missingDDMProxies(missingDDMProxies ~= curr_processes(i).DDMProxy);
                end

                % add default parameters
                offsetLength = length(processes);
                for i = 1:length(missingDDMProxies)

                    DDMParam = [];
                    switch(missingDDMProxies(i))
                        case EVC.DDM.DDMProc.DRIFT
                            DDMParam = defaultParams.drift;
                        case EVC.DDM.DDMProc.THRESH
                            DDMParam = defaultParams.thresh;
                        case EVC.DDM.DDMProc.BIAS
                            DDMParam = defaultParams.bias;
                        case EVC.DDM.DDMProc.NOISE
                            DDMParam = defaultParams.c;
                        case EVC.DDM.DDMProc.T0
                            DDMParam = defaultParams.T0;  
                    end

                    params{1} = DDMParam;
                    defaultVal = EVC.EVCFnc(EVC.EVCFnc.VALUE, params);
                    
                    defaultDuration = EVC.EVCFnc(EVC.EVCFnc.VALUE, {0});
                    
                    new_stage = curr_stage;

                    processes(offsetLength+i) = EVC.MSDDM.MSDDMProc(EVC.DDM.DDMProc.DEFAULT, missingDDMProxies(i), defaultVal, new_stage, defaultDuration);

                end
            
            end
            
            out = processes;
        end
        
    end

    
end

