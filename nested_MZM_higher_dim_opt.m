%% ==============================================================
% Nested MZM Higher-Dimensional Optimization
% Alexander Tiu
% 7/20/2026
%% ==============================================================

clc;
clear;
close all;

%% Step 1: Define the pilot experiment

% Qudit dimension.
D = 8;

% Begin with one nontrivial MUB.
% This will eventually range from MUB 1 through MUB 8.
mub_number = 1;

% Begin by optimizing only one column of the selected MUB.
% After the pilot works, this will eventually range from 1 through D.
column_number = 1;

% Random seed used to initialize particle swarm.
% Holding all other settings fixed while changing this value will allow
% us to study PSO repeatability later.
seed = 1;

% Set MATLAB's random-number generator before creating the swarm.
rng(seed, 'twister');

%% Step 2: Calculate the nested-MZM model size

% A D-dimensional implementation requires RF tones 1 through D - 1.
num_tones = D - 1;

% Each branch contains:
%   4(D - 1) modulation depths
%   4(D - 1) RF phases
%   2 bias phases
numVars = 8*num_tones + 2;

% Confirm the values expected for the present D = 8 experiment.
fprintf('Dimension: D = %d\n', D);
fprintf('RF tones per arm: %d\n', num_tones);
fprintf('Optimization parameters per branch: %d\n', numVars);
fprintf('Selected MUB: %d\n', mub_number);
fprintf('Selected column: %d\n', column_number);
fprintf('Random seed: %d\n', seed);