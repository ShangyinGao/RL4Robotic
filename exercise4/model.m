function [nextState,reward] = model(state,action)
    nextState = state + action;
    if (nextState <= 12 && nextState >= 1)
        
    else
        nextState = state;
    end
    
    if (sum(state == [1, 5, 9]) && action == -1)
        nextState = state;        
    end
    
    if (sum(state == [4, 8, 12]) && action == 1)
        nextState = state;
    end
    
    if (sum(state == [1, 2, 3, 4]) && action == -4)
        nextState = state;
    end
    
   if (sum(state == [9, 10, 11, 12]) && action == 4)
        nextState = state;
    end 
    
    if (state == 8)
        reward = 10;
    elseif (state == 12)
        reward = -10;
    else
        reward = -1;
    end   
end

