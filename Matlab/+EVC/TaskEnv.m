classdef TaskEnv < handle
    
    properties
        Sequence;           % trial sequence
        CurrTrialIdx        % current trialidx
        trialTypes          % all trial types (except choice trials)
    end
    
    methods
        
        % constructor
        % 1st argument: EVC.TaskEnv or EVC.Trial (single or list of trials)
        % 2nd argument: number of trials
        % 3rd argument: trial proportions
        % 4th argument: shuffle (1 = yes, 0 = no)
        function this = TaskEnv(Obj, varargin)
           
            % assign values
            if(isa(Obj, 'EVC.Trial') == 0 && isa(Obj, 'EVC.TaskEnv') == 0)
                error('Task Sequence needs to be an instance from class ''Task'' or ''TaskEnv''.');
            end
            
            if(isa(Obj, 'EVC.Trial'))
                this.Sequence = Obj;
                this.CurrTrialIdx = 1;
            end
            
            if(isa(Obj, 'EVC.TaskEnv'))
                this.Sequence = Obj.Sequence;
                this.CurrTrialIdx = Obj.CurrTrialIdx;
            end
            
            % check for additonal input arguments
            if(~isempty(varargin))
                
                    prob = [];
                    shuffle = 1;

                    % number of trials
                    if(length(varargin) >= 1)
                       if(~isa(varargin{1}, 'numeric'))
                          error('Second input argument should indicate number of trials (numeric).'); 
                       end
                       ntrials = varargin{1}; 
                    end

                    % trial proportions
                    if(length(varargin) >= 2)
                       if(~isa(varargin{2}, 'numeric'))
                          error('Third input argument should indicate trial proportions.'); 
                       end
                       if(length(varargin{2}) ~= length(this.Sequence))
                           error('Length of trial proportion array should match length of trial sequence');
                       end
                       prop = varargin{2}; 
                    end

                    % shuffle sequence?
                    if(length(varargin) >= 3)
                       if(~isa(varargin{3}, 'numeric') && ~isa(varargin{3}, 'bool'))
                          error('Fourth input argument should indicate whether sequence should be shuffled.'); 
                       end
                       shuffle = varargin{3}; 
                    end

                    if(isempty(prob)) prob = ones(length(this.Sequence)); end

                    % normalize proportions
                    prob = prob/sum(prob);

                    % fill sequence
                    counter = 1;
                    for i = 1:length(this.Sequence)
                        partialn = round(ntrials*prob(i));
                        for j = 1:partialn
                            newSeq(counter) = EVC.Trial(this.Sequence(i));
                            counter = counter+1;
                        end
                    end

                    this.Sequence = newSeq;
                    
                    if(shuffle)
                       this.shuffleSeq(); 
                    end
                
            end
            
            % get all cue types
            allCueIDs = [];
            this.trialTypes = {};
            for (i = 1:length(this.Sequence))
                % is it a non-choice trial?
                if(length(this.Sequence(i).ID) == 1)
                    % already registered this cueID?
                    if(length(allCueIDs) >= this.Sequence(i).cueID)
                        % already registered this ID?
                        if(~ismember(this.Sequence(i).ID, allCueIDs(this.Sequence(i).cueID).IDs))
                            allCueIDs(this.Sequence(i).cueID).IDs = [allCueIDs(this.Sequence(i).cueID).IDs this.Sequence(i).ID];
                            this.trialTypes{this.Sequence(i).ID} = this.Sequence(i);
                        end
                    else
                        % if cueID not registered, then create new entry
                        allCueIDs(this.Sequence(i).cueID).IDs = this.Sequence(i).ID;
                        this.trialTypes{this.Sequence(i).ID} = this.Sequence(i);
                    end
                end
            end

            
            % specify choice trials
            for (i = 1:length(this.Sequence))
                % choice trial
                if(length(this.Sequence(i).ID) > 1)
                    choiceIDs = [];
                    toDelete = [];
                    % for all choices among cue IDs
                    for(j = 1:length(this.Sequence(i).cueID))
                        % figure out valid trial IDs for this cue ID
                        if(this.Sequence(i).cueID(j) <= length(allCueIDs))
                            commonIDs = intersect(allCueIDs(this.Sequence(i).cueID(j)).IDs, this.Sequence(i).ID);
                            if(length(commonIDs) >= 1)
                                % if more than one valid trialID -> sample
                                choiceIDs = [choiceIDs randsample(commonIDs,1)];
                            else 
                                % if no trial IDs for this cue ID, then no
                                % choice is possible -> delete cue ID
                                %this.Sequence(i).cueID(this.Sequence(i).cueID == this.Sequence(i).cueID(j)) = [];
                                toDelete = [toDelete this.Sequence(i).cueID(j)];
                            end
                        else 
                            % if no trial with this cue ID exists, delete
                            % cue ID
                             %this.Sequence(i).cueID(this.Sequence(i).cueID == this.Sequence(i).cueID(j)) = [];
                             toDelete = [toDelete this.Sequence(i).cueID(j)];
                        end
                    end
                    this.Sequence(i).ID = choiceIDs;
                    
                    for(k = 1:length(toDelete))
                        this.Sequence(i).cueID(this.Sequence(i).cueID == toDelete(k)) = [];
                    end
                    
                    % after all modifications, is it still a choice trial?
                    if(length(this.Sequence(i).ID) == 0)
                        error('Choice trial has no valid trial IDs to choose from');
                    end
                    
                    % if there is only one trial ID to choose from, then
                    % make this trial a cued trial
                    if(length(this.Sequence(i).ID) == 1)
                        this.Sequence(i) = this.trialTypes{this.Sequence(i).ID};
                        EVC.HelperFnc.warning('A choice trial just turned into a cued trial (only one possible trial ID to choose from).');
                    end
                end
            end 
        end
        
        % adds a trial to existing sequence
        function addTrial(this, Trial)
            if(isa(Trial, 'EVC.Trial') == 0)
                error('Task Sequence needs to be an instance from class ''Task''.');
            end            
            this.Sequence = [this.Sequence Trial];
        end
        
        % returns and switches to the next trial
        function nextTrl = nextTrial(this)
            nextTrl = this.Sequence(this.CurrTrialIdx);
            
            if((this.CurrTrialIdx + 1) > length(this.Sequence))
                EVC.HelperFnc.warning('Already reached last trial in the sequence.');
            else
                this.CurrTrialIdx = this.CurrTrialIdx + 1;
                nextTrl = this.Sequence(this.CurrTrialIdx);
            end
        end
        
        % returns the current trial
        function currTrl = currentTrial(this)
           currTrl = this.Sequence(this.CurrTrialIdx); 
        end
        
        % randomly rearranges the trial order in the sequence
        function shuffleSeq(this)
            this.Sequence = this.Sequence(:,randperm(size(this.Sequence,2)));
        end
        
        % display trial sequence
        function dispSeq(this)
            for i = 1:length(this.Sequence)
               disp([num2str(i) ' ' this.Sequence(i).descr]); 
            end
        end
        
        % display sequence statistics
        function dispSeqStats(this)
           trialDescr = unique({this.Sequence.descr});
           
           for i=1:length(trialDescr)
                count_string=sum(ismember({this.Sequence.descr},trialDescr(i)));
                disp([trialDescr{i} ': ' num2str(count_string)]);
           end

        end
        
    end
    
    % specific task constructor methods
    methods(Static)
        
        function TaskEnvironment = randomTaskSwitchDesign(trialTypes, nTrials)
            tasks = unique([trialTypes.cueID]); % alternate between how many tasks?
            
            for trialIdx = 1:nTrials
               % choose task
               task = randsample(tasks,1);
               
               % choose trial
               trial = randsample(find([trialTypes.cueID] == task),1);
               
               trialSeq(trialIdx) = EVC.Trial(trialTypes(trial));
               
               if(trialIdx > 1)
                 if(trialSeq(trialIdx).cueID == trialSeq(trialIdx-1).cueID)
                    transition = 0; 
                 else
                    transition = 1;
                 end
               else
                   transition = -1;
               end
               trialSeq(trialIdx).conditions = [trialSeq(trialIdx).conditions transition];
            end
            
               TaskEnvironment =  EVC.TaskEnv(trialSeq);
        end
        
        function TaskEnvironment = blockedTaskSwitchDesign(trialTypes, nTrials, varargin)

           % additional input: blocksize
           if(length(varargin) >= 1)
              blockSize =  varargin{1};
           else 
              blockSize = 1;
           end
           
           % additional input: trial props
           if(length(varargin) >= 2)
              trialProps = varargin{2};
              if(length(trialProps) ~= length(trialTypes))
                 error('Size of trial porpotion vector doesn''t match number of given trials.'); 
              end
              gcdVal = trialProps(1);
              for i = 2:length(trialTypes)
                  gcdVal = gcd(trialProps(i), gcdVal);
              end
              trialProps = trialProps / gcdVal;
           else
              trialProps = ones(1,length(trialTypes)); 
           end
           
           % check blocksize
           if(floor(nTrials/blockSize) ~= nTrials/blockSize)
              EVC.HelperFun.warning('Number of trials cannot be divided into equal blocks with the given blocksize.'); 
           end
           
           % alternate between how many tasks?
           tasks = unique([trialTypes.cueID]);
           taskOrder = repmat(tasks, 1, ceil(nTrials/blockSize));
           
           if(blockSize == 0)
              blockSize = ceil(nTrials/tasks);
           end
           
           % create random trial sequence for each task
           taskTrialSeq = {};
           for i = 1:length(tasks)
                % which trial belongs to which trial type?
                taskTrials = find([trialTypes.cueID] == tasks(i));
                
                rep = trialProps(taskTrials);
                index = zeros(1,sum(rep));
                index(cumsum([1 rep(1:end-1)])) = 1;
                index = cumsum(index);
                currTaskTrialSeq = taskTrials(index);
                
                currTaskTrialSeq = repmat(currTaskTrialSeq, 1, ceil(nTrials*sum(trialProps(taskTrials))/(sum(trialProps)*length(currTaskTrialSeq))));
                
                taskTrialSeq{i} = currTaskTrialSeq(randperm(length(currTaskTrialSeq)));
           end
           
           % build sequence
           trialIdx = 0;
           for i = 1:length(taskOrder)
              
               currTask = taskOrder(i);
               toGo = min(blockSize, nTrials-trialIdx);
               
               for j = 1:toGo
                  
                  trialIdx = trialIdx + 1;
                  
                  % randomly select a trial of the corresponding task
                  currTaskTrialSeq = taskTrialSeq{currTask};
                  sampleIdx = currTaskTrialSeq(1);
                  
                  % delete that sample idx
                  taskTrialSeq{currTask} = currTaskTrialSeq(2:end);
                  
                  % add trial to sequence
                  trialSeq(trialIdx) = EVC.Trial(trialTypes(sampleIdx));

                  % add transition information
                  if(j==1)
                      transition = 1; % switch trial
                  else
                      transition = 0; % repetition trial
                  end
                  trialSeq(trialIdx).conditions = [trialSeq(trialIdx).conditions transition];
               end
               
           end
           
           TaskEnvironment = EVC.TaskEnv(trialSeq); 
        end
        
    end
    
end