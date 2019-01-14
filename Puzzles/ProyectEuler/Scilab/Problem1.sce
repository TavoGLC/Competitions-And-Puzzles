clear
clc

function[remValue]=LocalReminder(xVal,yVal)
    
    remValue=xVal-fix(xVal./yVal).*yVal
    
endfunction

container=0

for k=1:1:1000-1

    cVal=k
    
    if LocalReminder(cVal,3)==0 | LocalReminder(cVal,5)==0
        
        container=container+cVal
        
    end
    
end

disp(container)
