clc; clear all; close all;

%% system parameter
% continous system
k = 1;
m = 1;
c = 1;
A_cont = [0, 1; -k/m, -c/m];
B_cont = [0; 1/m];

% time parameter
time_step = 0.01;
time_whold = 10;
num_seq = ceil(time_whold / time_step);

% discrete system
sysd = c2d(ss(A_cont, B_cont, [], []), time_step);
sys.A = sysd.A;
sys.B = sysd.B;

% state [postion; velocity]
state_init = [0; 0];
state_goal = [0.6; 0];
num_state = size(state_init, 1);
num_rollouts = 20;
num_rollouts_best = 10;
% states = zeros(num_state, num_seq, num_rollouts);
% states_2D = zeros(num_state, num_seq);
% rewards = zeros(1, num_rollouts);

% policy parameters = [weight, bias] weight 2 numbers, bias 1 number
par = zeros(1, 3);
pars = zeros(num_rollouts, 3);
par_new = zeros(1, 3);

% flag
policy_converge = false;
num_loop = 1;
num_loop_max = 100;

%% main function
par_hist = zeros(num_loop_max, 3);
par_norm_hist = zeros(1, num_loop_max);
while(~policy_converge)
    % parameter space exploration
    [states, pars] = exploration(par, state_init, num_seq, num_rollouts, sys);
    rewards = rewardCal(states, state_goal, num_seq, num_rollouts);
    par_new = updatePar(par, pars, rewards, num_rollouts_best);
    par_hist(num_loop, :) = par;
    par_norm_hist(num_loop) = norm(par_new - par);
    state_2D = nextStates(state_init, par, num_seq, sys);
    % converge criteria 
    if (norm(state_2D(:, end)-state_goal)<0.01)
        policy_converge = true;
    else
        par = par_new;
    end    
    disp(num_loop);
    disp(par);
    
    num_loop = num_loop + 1;
end

% visualize the result
x = 1:num_seq;
plot(x, state_2D(1, :), x, state_2D(2, :));



%% some functions
% return: 2-D states
function states = nextStates(state_init, par, num_seq, sys)
    % states = [num_state, num_seq]
    states = zeros(2, num_seq);
    states(:, 1) = state_init;
    for idx = 2:num_seq
        % use linear policy: u = weight * state + bias
        force = par * [states(:, idx-1); 1];
        states(:, idx) = sys.A * states(:, idx-1) + sys.B * force;
    end
end

% generate 20 rollouts (pars) and 
% calculate 3-D states (each layer stands for each rollouts)
% function [states, pars] = exploration(par, state_init, num_seq, num_rollouts)
function [states, pars] = exploration(par, state_init, num_seq, num_rollouts, sys)
    % pars = [num_rollouts, 3]
    pars = repmat(par, num_rollouts, 1) + randn(num_rollouts, size(par, 2));
    % 3-D states
    states = zeros(2, num_seq, num_rollouts);
    for i = 1:num_rollouts
        % states(:, :, i) = nextStates(state_init, pars(i, :), num_seq);
        % states = [num_state, num_seq]
        states(:, 1, i) = state_init;
        for j = [2:num_seq]
            % use linear policy: u = weight * state + bias
            force = pars(i, :) * [states(:, j-1, i); 1];
            states(:, j, i) = sys.A * states(:, j-1, i) + sys.B * force;
        end        
        
    end
end

% calculate reward for 20 rollouts
function rewards = rewardCal(states, goal, num_seq, num_rollouts)
    % states = [num_state, num_seq, num_rollouts]
    rewards = zeros(1, num_rollouts);
    for idx = 1:num_rollouts
        errors = repmat(goal, 1, num_seq) - states(:, :, idx);
        rewards(idx) = sum(exp(-vecnorm(errors).^2)) / num_seq;        
    end

end

% update parameter
function par_new = updatePar(par, pars, rewards, num_rollouts_best)
    % choose best 10 rollouts
    [~, idx_sort] = sort(rewards, 'descend');
    idx_best = idx_sort(1:num_rollouts_best);
    rewards_best = rewards(idx_best);
    pars_best = pars(idx_best, :);
    pars_best_diff = pars_best - repmat(par, num_rollouts_best, 1);
    par_new = par + rewards_best * pars_best_diff / sum(rewards_best);
    
end

