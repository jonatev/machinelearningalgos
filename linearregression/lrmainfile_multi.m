%% Linear regression with multiple variables
%
%  Instructions
%  ------------
%  The following functions are used:
%
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%

%% Initialization

%% ================ Part 1: Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];


%% ================ Part 2: Gradient Descent ================

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha1 = 0.1;
alpha2 = 0.03;
alpha3 = 0.01;
num_iters = 500;

% Init Theta and Run Gradient Descent 
theta1 = zeros(3, 1);
theta2 = zeros(3, 1);
theta3 = zeros(3, 1);
[theta1, J_history1] = gradientDescentMulti(X, y, theta1, alpha1, num_iters);
[theta2, J_history2] = gradientDescentMulti(X, y, theta2, alpha2, num_iters);
[theta3, J_history3] = gradientDescentMulti(X, y, theta3, alpha3, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history1), J_history1, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

hold on;
plot(1:numel(J_history3), J_history3, '-r', 'LineWidth', 2);
hold on;
plot(1:numel(J_history2), J_history2, '-k', 'LineWidth', 2);


% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta1, theta2, theta3);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.
price = 0; % You should change this

% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 3: Normal Equations ================

fprintf('Solving with normal equations...\n');

%% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');


% Estimate the price of a 1650 sq-ft, 3 br house
X1=[1650 3];

X_norm1 =eye(size(X1));
for i=1:size(X1,2)
    X_norm1(i)=(X1(i)-mu(i))./sigma(i);
    
price = theta'*[1; X_norm1']; % You should change this
end

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price);

