classdef NetworkDesign < matlab.mixin.Copyable
    %NETWORKDESIGN a deep net specification (ei in stanford tutorial)
    
    properties
        ei
    end
    
    methods
        function obj = NetworkDesign(ei)
            %We take multi-input for network spec
            for i=1:length(ei)
                
                %We have a multi-network design 
                if length(ei) > 1
                    if iscell(ei)
                        obj(i)=NetworkDesign(ei{i});
                    else
                        obj(i)=NetworkDesign(ei(i));
                    end
                    
                %We have a single network design
                else
                    obj.ei=ei;
                end
            end
        end   
    end
end
