classdef DDMProc < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        type                % ENUM: which type of process (control/actual state/expected state)?
        DDMProxy            % ENUM: to which DDM parameter does the process map?
        input               % EVCFnc: input to DDM parameter  
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
        t0 = 5;
    end
    
    methods
        
        function this = DDMProc(type, DDMProxy, input)
            
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
            
        end
        
        % returns outcome value
        function out = getVal(this)
           out = this.input.getVal(); 
        end
        
    end
    
    methods (Static = true)
        
        % return processes with specified process types
        function out = filterProcess(processes, varargin)
            
            if(~isa(processes, 'EVC.DDM.DDMProc'))
               error('First input argument needs to be an instance of EVC.DDM.DDMProcess.'); 
            end
            
            % process input
            allowedTypes = [];
            typeIndex = 1;
            for i = 1:length(varargin)
               if(isnumeric(varargin{i}))
                   allowedTypes(typeIndex) = varargin{i};
                   typeIndex = typeIndex + 1;
               end
            end
            
            
            out = EVC.DDM.DDMProc.empty(1,0);
            outIndex = 1;
            
            % only add processes with allowed process types
            for i = 1:length(processes)
                % true if at least one of the process types is a member of allowed types
                if(ismember(true,ismember(processes(i).type, allowedTypes)))
                   out(outIndex) = processes(i);
                   outIndex = outIndex + 1;
                end
            end
            
        end
        
        % add default processes for not-specified DDM processes
        function out = addDefaultParams(processes, defaultParams)
            
            if(~isa(processes, 'EVC.DDM.DDMProc'))
               error('First input argument needs to be an instance of EVC.DDM.DDMProcess.'); 
            end
            
            % which DDM parameters aren't specified yet?
            missingDDMProxies = [EVC.DDM.DDMProc.DRIFT EVC.DDM.DDMProc.THRESH EVC.DDM.DDMProc.BIAS EVC.DDM.DDMProc.NOISE EVC.DDM.DDMProc.t0];
            for i = 1:length(processes)
                missingDDMProxies = missingDDMProxies(missingDDMProxies ~= processes(i).DDMProxy);
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
                    case EVC.DDM.DDMProc.t0
                        DDMParam = defaultParams.t0;
                end
                
                params{1} = DDMParam;
                defaultVal = EVC.EVCFnc(EVC.EVCFnc.VALUE, params);
                processes(offsetLength+i) = EVC.DDM.DDMProc(EVC.DDM.DDMProc.DEFAULT, missingDDMProxies(i), defaultVal);
                
            end
            
            out = processes;
        end
    end

    
end

