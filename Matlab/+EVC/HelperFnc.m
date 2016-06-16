classdef HelperFnc

properties
    
end

%% Helper functions

    methods (Static = true)
       
        function out = ifelse(bool, out1, out2)
           if(bool)
               out = out1; 
           else
               out = out2;
           end
        end
        
        function disp(source, varargin)
           import EVC.*;
           config;
           
           if(diagnosis)

               if(eval(strcat('Diagnosis.',source)))
                  disp(strcat('----------------', source, ':'));
                  for i = 1:length(varargin)
                      disp(varargin{i});
                  end
               end
           
           end
           
        end
        
        function warning(varargin)
           import EVC.*;
           config;
           
           if(warnings)
              for i = 1:length(varargin)
                 warning(varargin); 
              end
           end
        end
        
        function baseStruct = addStruct(varargin) % only works for 1 level
           
           if(isempty(varargin))
               return;
           end
           
           % extract weight
           weight = 1*ones(length(varargin)-1);
           for i = 1:length(varargin)
               if(isnumeric(varargin{i}))
                    weight = varargin{i};
               end
           end
            
           baseStruct = varargin{1};
           baseFields = fieldnames(baseStruct);
           % for all remaining structures
           if(length(varargin) > 1)
                for i = 2:length(varargin)
                    if(isstruct(varargin{i}))
                        % get the fieldnames
                        fields = fieldnames(varargin{i});
                        for j = 1:length(fields)
                            % add fields if the corresponding field is shared
                            if(ismember(fields(j), baseFields))
                                baseField = '';
                                eval(strcat('baseField = baseStruct.',fields{j},';'));
                                addField = eval(strcat('varargin{',num2str(i),'}.',fields{j}, ';'));

                                info_baseField = whos('baseField');
                                info_addField = whos('addField');

                                if(isequal(info_baseField.class, info_addField.class))
                                    baseField = baseField + addField .* weight(i-1);
%                                     disp('here');
                                end
                                eval(strcat('baseStruct.', fields{j},'= baseField;'));
                            % if field not shared, then add to structure
                            else
                               eval(strcat('baseStruct.', fields{j},'=','varargin{',num2str(i),'}.', fields{j},';'));
                            end
                        end
                    end
                end
           end
           
        end
        
        function baseStruct = weightStruct(baseStruct, weight) % only works for 1 level
           
           baseFields = fieldnames(baseStruct);
           for i = 1:length(baseFields)
               eval(strcat('baseField = baseStruct.',baseFields{i},';'));
               eval(strcat('baseStruct.', baseFields{i},'= baseField*weight;'));
           end
           
        end
        
    end

end