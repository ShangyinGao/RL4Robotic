% 2017-11-15
% Shangyin Gao

% % Exercise 1
P = [0.5, 0.5, 0; 0.2, 0.1, 0.7; 0, 0.9, 0.1];
R = [0; 10; 0];
gamma = 0.9;
V = inv(eye(3)-gamma*P)*R;

[Max, idx] = max(V);
disp(idx)

% % Exercise 2
% value interation
% row number of the world
wRow = 2;
% column number of the world
wCol = 2;
% number of action
nAct = 4;
V = zeros(wRow*wCol, 1);
reward = 10;
% Probability of action (GL, GR, GU, GD)
ProbAct = [0.8, 0.4, 0.9, 0.8];
% transition probability init
Prob = zeros(4, 4, 4);
% reward
R = zeros(wRow*wCol, 1);
% number encode method: from left to right and from top to bottom
R((wRow-1)*wCol) = reward;
R(wRow*wCol) = -reward;

% % for loop to init Prob matrix is too difficult
% for l = 1:nAct % layer
%     for i = 1: wRow % row
%         for j = 1:wCol % column
%             if i == 1 % wall on top
%                 if j == 1 % wall on left
%                     Prob(i, j, l) = 
%                 elseif j == wCol % wall on right
%                     Prob(i, j, l) = 
%                 
%             elseif i == wRow % wall on bottom
%                 if j == 1 % wall on left
%                     Prob(i, j, l) = 
%                 elseif j == wCol % wall on right                
%                     Prob(i, j, l) = 

Prob(:,:,1) = [0.05, 0.05, 0.05, 0.05;
               0.8, 0.2/3, 0.2/3, 0.2/3;
               0.05, 0.05, 0.05, 0.05;
               0.2/3, 0.2/3, 0.8 0.2/3];
           
Prob(:,:,2) = [0.2, 0.4, 0.2 ,0.2;
               0.6/4, 0.6/4, 0.6/4, 0.6/4;
               0.2, 0.2, 0.2, 0.4;
               0.6/4, 0.6/4, 0.6/4, 0.6/4];         
    
Prob(:,:,3) = [0.1/4, 0.1/4, 0.1/4, 0.1/4;
               0.1/4, 0.1/4, 0.1/4, 0.1/4;
               0.9, 0.1/3, 0.1/3, 0.1/3;
               0.1/3, 0.9, 0.1/3, 0.1/3];  
           
Prob(:,:,4) = [0.2/3, 0.2/3, 0.8, 0.2/3;
               0.2/3, 0.2/3, 0.2/3, 0.8;
               0.2/4, 0.2/4, 0.2/4, 0.2/4;
               0.2/4, 0.2/4, 0.2/4, 0.2/4;];       
    
% threshold
threshold = 0.1;
% max number of loop
Lmax = 100;
for i = 1:Lmax
    while(abs(Vnew - Vnew)>threshold)
        % build the new matrix
        tmpMatrix = [Prob(:,:,1)*V, Prob(:,:,2)*V, Prob(:,:,3)*V, Prob(:,:,4)*V]
        Vnew = R + gamma*max()



