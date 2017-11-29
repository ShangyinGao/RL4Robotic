clc; clear all; close all;

%% System prefereces & parameters
% System constants:
sys.k = 1;
sys.m = 1;
sys.c = 0;
sys.A = [0, 1; -sys.k/sys.m, -sys.c/sys.m];
sys.B = [0; 1/sys.m];

% Q-agent preference:
sys.max = 5;   % Q mapping space width
sys.res = 0.1;  % Q mapping resolution
sys.domain.p = -sys.max:sys.res:sys.max;    % Q mapping space for position [-10, 10]
sys.domain.v = -sys.max:sys.res:sys.max;    % Q mapping space for velocity [-10, 10]
sys.domain.f = -1.5:0.01:1.5;                 % Q mapping space for action(force) [0, 1]

% Noise setting:
sys.noiseScale = 0.0; % Noise scale factor. Appropriate noise should not exceed 0.1

% Adams-Bashforth coefficients:
% sys.coef = cell(1,5);
% sys.coef{1} = 1;
% sys.coef{2} = [3/2 -1/2];
% sys.coef{3} = [23/12 -4/3 5/12];
% sys.coef{4} = [55/24 -59/24 37/24 -3/8];
% sys.coef{5} = [1901/720 -1387/360 109/30 -637/360 251/720];

% System delta time and target timer:
sys.targetSec = 5;  % Terminating seconds
sys.dt = 0.025;     % cart system simulation time interval
sys.targetSeq = ceil(sys.targetSec/sys.dt);
sys.tMax        = 120;

% Target position and velocity:
sys.targetX = [0.6; 0]; % p = 0.6m, v = 0m/s

% Q learning preference:
sys.eps = 0.05;          % const for epsilon greedy
sys.alpha = 0.15;        % learning rate
sys.gamma = 0.99;        % discount factor
sys.Q = 0.01*randn(length(sys.domain.p), length(sys.domain.v), length(sys.domain.f));    % Q mapping matrix

%% Main Script

% Training by running episodes.
plt = figure('Name', 'Training Episode');
totalSuccess = 0;
episode = 1;
while totalSuccess < 5000
    display(['Episode: ', num2str(episode)]);
    [history, sys, result] = runEpisode(sys);
    totalSuccess = totalSuccess + result;
    clf(plt);
    hold on;
    plot(history.t, history.f, 'r');
    plot(history.t, history.X(2,:), 'b');
    plot(history.t, history.X(1,:), 'k');
    line([0 120], [sys.targetX(1) sys.targetX(1)], 'LineStyle', '--');
    xlim([0 history.t(end)]);
    ylim([-0.5*sys.max 0.5*sys.max]);
    title(['Episode: ', num2str(episode)]);
    xlabel('t(second)');
    grid on;
    hold off;
    
    drawnow;
    episode = episode + 1;
end

% Run test episode.
for test = 1:10
    plt = figure('Name', 'Test Episode');
    display(['Episode: ', num2str(test)]);
    history = runTest(sys);

    clf(plt);
    hold on;
    plot(history.t, history.f, 'r');
    plot(history.t, history.X(2,:), 'b');
    plot(history.t, history.X(1,:), 'k');
    line([0 120], [sys.targetX(1) sys.targetX(1)], 'LineStyle', '--');
    title(['Test Episode: ', num2str(test)]);
    xlim([0 120]);
    ylim([-0.5*sys.max 0.5*sys.max]);
    xlabel('t(second)');
    grid on;
    hold off;
    drawnow;
end
%% Functinos...
% Run a test episode (non-greedy search)
function history = runTest(sys)
    isTerminated = 0;
    
    X = [0.5*sys.max*rand;0];
    t = 0;
    
    history.X   = [];
    history.dX  = [];
    history.t   = [];
    history.f   = [];
    
    while ~isTerminated
        state   = getState(X, sys);
        policy  = getPolicy(state, sys, 0);
        force   = getForce(policy, sys);
        
        dX = getDiff(X, force, sys);
        
        history.X   = [history.X, X];
        history.dX 	= [history.dX, dX];
        history.t   = [history.t, t];
        history.f   = [history.f, force];
        
        nextX = evolveX(history, sys); 
        
        if nextX(1) < sys.domain.p(1) || nextX(1) > sys.domain.p(end)
            fprintf('Episode met stop criterion: Out of position range\n');
            isTerminated = 1; 
        end
        if nextX(2) < sys.domain.v(1) || nextX(2) > sys.domain.v(end)
            fprintf('Episode met stop criterion: Out of velocity range\n');
            isTerminated = 1;
        end
        
        X = nextX;
        t = t + sys.dt;
        
        if t >= sys.tMax
           isTerminated = 1; 
        end
    end
end

% Run a training episode (eps-greedy search)
function [history, sys, result] = runEpisode(sys)
    isTerminated = 0;
    
    X = [0.5*sys.max*rand; 0];
    t = 0;
    targetState = getState(sys.targetX, sys);
    stopSeq = 0;
    history.X   = []; % history of evolution of X, dX, force and timeline
    history.dX  = [];
    history.t   = [];
    history.f   = [];
    history.s.p = [];
    history.s.v = [];
    history.p   = [];
   
    
    while ~isTerminated
        
        [state, history] = recordHistory(X, t, history, sys);
        sys = updateQ(history, sys);
        
        nextX = evolveX(history, sys);
        
        % throw an error if X goes out of agent's Q mapping space and
        % terminate iteration
        if nextX(1) < sys.domain.p(1) || nextX(1) > sys.domain.p(end) 
            fprintf('Episode met stop criterion: Out of position range\n');
            result = 0;
            isTerminated = 1; 
        end
        if nextX(2) < sys.domain.v(1) || nextX(2) > sys.domain.v(end)
            fprintf('Episode met stop criterion: Out of velocity range\n');
            result = 0;
            isTerminated = 1;
        end
        
        X = nextX;
        t = t + sys.dt;
        
        % Stopping criterion: Target Reached. 
        % Terminate iteration if target position & velocity is satisfied
        % for desired seconds in a row
        if state.p == targetState.p && state.v == targetState.v
            stopSeq = stopSeq + 1;
            if stopSeq == sys.targetSeq
                fprintf('Episode met stop criterion: Reached target X\n');
                result = 1;
                isTerminated = 1;
            end
        else
            stopSeq = 0;
        end
        % Stopping criterion: Time out
        if t >= 5*sys.tMax
            fprintf('Episode ran out of time(%3ds)\n', 5*sys.tMax);
            result = 0;
            isTerminated = 1;
        end
            
    
    end
end

function [state, history] = recordHistory(X, t, history, sys)
    state   = getState(X, sys);
    policy  = getPolicy(state, sys, 1);
    force   = getForce(policy, sys);
    dX = getDiff(X, force, sys);

    history.X   = [history.X, X];
    history.dX 	= [history.dX, dX];
    history.s.p   = [history.s.p, state.p];
    history.s.v   = [history.s.v, state.v];
    history.t   = [history.t, t];
    history.f   = [history.f, force];
    history.p   = [history.p, policy];
end

function sys = updateQ(history, sys)
    if length(history.t) <= 1
        currentState.p = history.s.p(end);
        currentState.v = history.s.v(end);
        reward = getReward(currentState, sys);
        sys.Q(currentState.p, currentState.v, history.p(1, end)) = ...
            (1-sys.alpha)*sys.Q(currentState.p, currentState.v, history.p(1, end)) ...
            + sys.alpha*reward;
    else
        currentState.p = history.s.p(end);
        currentState.v = history.s.v(end);
        prevState.p = history.s.p(end-1);
        prevState.v = history.s.v(end-1);
        reward = getReward(prevState, sys);
        
        sys.Q(prevState.p, prevState.v, history.p(1, end-1)) = ...
            (1-sys.alpha)*sys.Q(prevState.p, prevState.v, history.p(1, end-1)) ...
            + sys.alpha*(reward + sys.gamma*max(sys.Q(currentState.p, currentState.v, :)));
    end
end

% Translate X to state in Q matrix (Discretizing)
function state = getState(X, sys)
    noiseX = X + sys.noiseScale*sys.res*randn(2,1);
    [~, state.p] = min(abs(sys.domain.p - noiseX(1)));
    [~, state.v] = min(abs(sys.domain.v - noiseX(2)));
end

% Get a policy (eps-greedy or non-greedy)
function policy = getPolicy(state, sys, epsGreedy)
    if epsGreedy
        if rand > sys.eps
           [~, policy] = max(sys.Q(state.p, state.v, :)); 
        else
            policy = ceil(length(sys.domain.f)*rand);
        end
    else
        [~, policy] = max(sys.Q(state.p, state.v, :));
    end
    
end

% Get force referring to policy
function force = getForce(policy, sys)
    force = sys.domain.f(policy);
end

% Get dX
function dX = getDiff(X, force, sys)
    dX = sys.A*X + sys.B*force;
end

% Evolution of X
function nextX = evolveX(history, sys)
    %  Euler method
    history.dX(1,end) = history.X(2,end) + sys.dt*history.dX(2,end);

    nextX = history.X(:, end) + sys.dt*history.dX(:,end);
    % AB 5th method
    %     step = size(history.dX, 2); % 
    %     if step > 5
    %         step = 5;
    %     end
    %     nextX = history.X(:, end) + sys.dt*sum(sys.coef{step}.*history.dX(:, end-step+1:end), 2);
end

% Get reward regarding state in Q
function reward = getReward(state, sys)
    [~, target.p] = min(abs(sys.domain.p - sys.targetX(1)));
    [~, target.v] = min(abs(sys.domain.v - sys.targetX(2)));
        
    if target.p < state.p
        if sys.domain.v(state.v) > 0
            reward = -1;
        else
            reward = 1;
        end
    else
        if sys.domain.v(state.v) > 0
            reward = 1;
        else
            reward = -1;
        end
    end
    
    if target.p == state.p && target.v == state.v
        reward = 100;
    elseif state.p == 1 || state.p == length(sys.domain.p)
        reward = -100;
    elseif state.v == 1 || state.v == length(sys.domain.v)
        reward = -100;
    end
end

