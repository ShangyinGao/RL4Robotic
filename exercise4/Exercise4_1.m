alpha = 0.1;
gamma = 0.7;
epoch = 100000;
epsilon = 0.8;
% way to define the state number
% leftTOP = 1, rightDown = 12;
% from left to right, from top to down
state = [1:1:12];
% action: left, right, up, down
action = [-1, 1, -4, 4];
% state_idx = 10;
Q = zeros(length(state), length(action));


for epoch = 1:epoch
    state_idx = randi(length(state));
    r = rand;
    x = sum(r >= [0, 1-epsilon, epsilon]);
    if x == 1   
        [~, umax] = max(Q(state_idx,:));
        currentAction = action(umax);
    else        
        currentAction = datasample(action,1); 
    end

%     currentAction = max(Q(state_idx,:));
    action_idx = find(action==currentAction);
    [nextState,nextReward] = model(state(state_idx),action(action_idx));
    next_state_idx = find(state==nextState);
    Q(state_idx,action_idx) = Q(state_idx,action_idx) + alpha * (nextReward + gamma* max(Q(next_state_idx,:)) ...
        - Q(state_idx,action_idx));
    
    if (next_state_idx == 12 || next_state_idx == 6)
        state_idx = datasample(2:length(state)-1,1); 
    else
        state_idx = next_state_idx;
    end
    
end

[~, choice]=max(Q,[],2);                             
displayMatrix = displayAction(choice);
disp("optimal policy: ");
disp(displayMatrix);