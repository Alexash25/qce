%% ==============================================================
% Nested MZM Higher-Dimensional Seed Sweep
% Alexander Tiu
% 7/20/2026
%
% This script:
%   1. Optimizes every nontrivial D-dimensional MUB gate.
%   2. Optimizes every column of each MUB gate.
%   3. Repeats the complete experiment across several PSO seeds.
%   4. Calculates the mean, standard deviation, and standard error.
%   5. Displays fidelity and runtime graphs.
%
% No results or figures are saved to disk.
%% ==============================================================

clc;
clear;
close all;


%% Step 1: Define the seed-sweep experiment

% Qudit dimension.
D = 8;

% The nontrivial MUB gates are U1 through UD.
mub_numbers = 1:D;

% Every gate contains D columns.
column_numbers = 1:D;

% PSO seeds included in the statistical experiment.
%
% Begin with 20 seeds.
% Change this to 1:42 for the larger final sweep.
seed_values = 1:20;

% Total number of independent seeded experiments.
num_seeds = numel(seed_values);

% Select which uncertainty measure appears on the graphs.
%
% Available choices:
%   'standard deviation'
%   'standard error'
error_bar_type = 'standard deviation';


%% Step 2: Calculate the nested-MZM model size

% A D-dimensional implementation requires RF tones 1 through D - 1.
num_tones = D - 1;

% Each branch contains:
%   4(D - 1) modulation depths
%   4(D - 1) RF phases
%   2 bias phases
numVars = 8*num_tones + 2;

fprintf('====================================================\n');
fprintf('D = %d PSO SEED-SWEEP EXPERIMENT\n', D);
fprintf('====================================================\n');
fprintf('RF tones per arm: %d\n', num_tones);
fprintf('Optimization parameters per branch: %d\n', numVars);
fprintf('MUB gates: U1 through U%d\n', D);
fprintf('Columns per MUB gate: %d\n', D);
fprintf('Number of seeds: %d\n', num_seeds);
fprintf('Seed values: %s\n', mat2str(seed_values));
fprintf('Total PSO calls: %d\n', num_seeds * D * D);
fprintf('====================================================\n');


%% Step 3: Load the D-dimensional MUB matrices

% Find the folder containing this optimizer script.
script_dir = fileparts(mfilename('fullpath'));

% Build the complete path to the MUB data file.
mub_file = fullfile( ...
    script_dir, ...
    sprintf('mubs_D%d.mat', D));

% Confirm that the file exists.
if ~isfile(mub_file)

    error( ...
        'Could not find %s.', ...
        mub_file);

end

% Load the variables stored in the MUB file.
loaded_data = load(mub_file);

% Confirm that the expected MUB array exists.
if ~isfield(loaded_data, 'U_mubs')

    error( ...
        '%s does not contain a variable named U_mubs.', ...
        mub_file);

end

U_mubs = loaded_data.U_mubs;

% A complete MUB set contains D + 1 bases.
expected_size = [D, D, D + 1];

if ~isequal(size(U_mubs), expected_size)

    error( ...
        'Expected U_mubs to have size %d x %d x %d, but received %s.', ...
        D, ...
        D, ...
        D + 1, ...
        mat2str(size(U_mubs)));

end

fprintf('\nLoaded MUB file:\n%s\n', mub_file);


%% Step 4: Prepare and verify all nontrivial target matrices

% Store U1 dagger through UD dagger.
target_matrices_all = complex(zeros(D, D, D));

% Store the unitarity error of every target matrix.
unitarity_errors = zeros(D, 1);

for mub_number = mub_numbers

    % The stored basis at index 1 is U0.
    basis_index = mub_number + 1;

    % Select Um.
    U_m = U_mubs(:, :, basis_index);

    % Measurement requires Um dagger.
    U_target = U_m';

    % Store the target matrix.
    target_matrices_all(:, :, mub_number) = U_target;

    % Verify unitarity.
    unitarity_errors(mub_number) = ...
        norm(U_m' * U_m - eye(D), 'fro');

    if unitarity_errors(mub_number) > 1e-8

        error( ...
            'MUB %d failed the unitarity check.', ...
            mub_number);

    end

end

fprintf('\nAll target matrices passed the unitarity check.\n');
fprintf('Maximum unitarity error: %.3e\n', ...
    max(unitarity_errors));


%% Step 5: Define the complete simulated frequency range

% Highest Bessel order retained for each RF tone.
max_bessel_order = 4;

% Maximum possible output-frequency order.
max_frequency_order = ...
    max_bessel_order * sum(1:num_tones);

% Complete simulated frequency-order range.
order_range = ...
    -max_frequency_order:max_frequency_order;

fprintf('\nSimulated frequency-order range: %d through %d\n', ...
    order_range(1), ...
    order_range(end));


%% Step 6: Define the nested-MZM parameter bounds

num_beta_parameters = 4 * num_tones;
num_phase_parameters = 4 * num_tones;

% Modulation depths are restricted to [0, 2].
beta_lower_bounds = ...
    zeros(1, num_beta_parameters);

beta_upper_bounds = ...
    2 * ones(1, num_beta_parameters);

% RF phases are restricted to [-pi, pi].
phase_lower_bounds = ...
    -pi * ones(1, num_phase_parameters);

phase_upper_bounds = ...
     pi * ones(1, num_phase_parameters);

% Two nested-MZM bias phases use the same range.
bias_lower_bounds = [-pi, -pi];
bias_upper_bounds = [ pi,  pi];

% Combine all parameter bounds.
lb = [ ...
    beta_lower_bounds, ...
    phase_lower_bounds, ...
    bias_lower_bounds ...
    ];

ub = [ ...
    beta_upper_bounds, ...
    phase_upper_bounds, ...
    bias_upper_bounds ...
    ];

% Verify the bound-vector sizes.
if numel(lb) ~= numVars || numel(ub) ~= numVars

    error( ...
        'The lower and upper bounds must each contain %d values.', ...
        numVars);

end

fprintf('\nParameter bounds prepared:\n');
fprintf('Modulation-depth parameters: %d\n', ...
    num_beta_parameters);
fprintf('RF-phase parameters: %d\n', ...
    num_phase_parameters);
fprintf('Bias-phase parameters: 2\n');
fprintf('Total parameters: %d\n', numVars);


%% Step 7: Configure particle swarm

pilot_swarm_size = 100;
pilot_max_iterations = 100;
pilot_max_stall_iterations = 30;

options = optimoptions( ...
    'particleswarm', ...
    'CreationFcn', @pswcreationuniform, ...
    'Display', 'off', ...
    'FunctionTolerance', 1e-6, ...
    'InertiaRange', [0.5, 1.5], ...
    'MaxIterations', pilot_max_iterations, ...
    'MaxStallIterations', pilot_max_stall_iterations, ...
    'SwarmSize', pilot_swarm_size, ...
    'UseParallel', false, ...
    'UseVectorized', false);

fprintf('\nPSO settings:\n');
fprintf('Swarm size: %d particles\n', ...
    pilot_swarm_size);
fprintf('Maximum iterations: %d\n', ...
    pilot_max_iterations);
fprintf('Maximum stall iterations: %d\n', ...
    pilot_max_stall_iterations);
fprintf('Parallel processing enabled: No\n');


%% Step 8: Prepare seed-sweep storage

% Dimensions:
%   first index  = seed
%   second index = MUB gate
gate_fidelities_by_seed = ...
    zeros(num_seeds, D);

gate_leakages_by_seed = ...
    zeros(num_seeds, D);

gate_costs_by_seed = ...
    zeros(num_seeds, D);

gate_runtimes_by_seed = ...
    zeros(num_seeds, D);

minimum_column_fidelities_by_seed = ...
    zeros(num_seeds, D);

maximum_column_fidelities_by_seed = ...
    zeros(num_seeds, D);


% Dimensions:
%   first index  = seed
%   second index = MUB gate
%   third index  = column
column_fidelities_by_seed = ...
    zeros(num_seeds, D, D);

column_leakages_by_seed = ...
    zeros(num_seeds, D, D);

column_costs_by_seed = ...
    zeros(num_seeds, D, D);

column_runtimes_by_seed = ...
    zeros(num_seeds, D, D);

column_iterations_by_seed = ...
    zeros(num_seeds, D, D);

column_function_evaluations_by_seed = ...
    zeros(num_seeds, D, D);

column_exitflags_by_seed = ...
    zeros(num_seeds, D, D);


% Wall-clock runtime of each complete seed.
seed_wall_runtimes = zeros(num_seeds, 1);


%% Step 9: Begin the complete seed sweep

complete_experiment_start = tic;

for seed_index = 1:num_seeds

    seed = seed_values(seed_index);

    % Reset MATLAB's random-number generator for this seed.
    rng(seed, 'twister');

    seed_start = tic;

    fprintf('\n\n');
    fprintf('####################################################\n');
    fprintf('STARTING SEED %d\n', seed);
    fprintf('Seed %d of %d\n', seed_index, num_seeds);
    fprintf('####################################################\n');


    %% Step 10: Optimize every nontrivial MUB for this seed

    for mub_number = mub_numbers

        U_target = ...
            target_matrices_all(:, :, mub_number);

        fprintf('\n');
        fprintf('----------------------------------------------------\n');
        fprintf('Seed %d: Optimizing U%d dagger\n', ...
            seed, ...
            mub_number);
        fprintf('----------------------------------------------------\n');


        % Temporary storage for this complete MUB gate.
        current_column_fidelities = zeros(1, D);
        current_column_leakages = zeros(1, D);
        current_column_costs = zeros(1, D);
        current_column_runtimes = zeros(1, D);

        current_column_iterations = zeros(1, D);
        current_column_function_evaluations = zeros(1, D);
        current_column_exitflags = zeros(1, D);


        %% Step 11: Optimize every column of the current MUB

        for column_number = column_numbers

            % Extract the target column.
            target_column = ...
                U_target(:, column_number);

            target_amplitudes = ...
                abs(target_column).';

            target_phases = ...
                angle(target_column).';

            % Shift the output orders according to the input column.
            target_orders = ...
                (0:D-1) - (column_number - 1);


            %% Step 12: Create the objective function

            cost_fun = @(x) ...
                nested_MZM_higher_dim_cost( ...
                    x, ...
                    target_orders, ...
                    target_amplitudes, ...
                    target_phases, ...
                    order_range, ...
                    D);


            %% Step 13: Run PSO for the current column

            optimization_start = tic;

            [x_best, ~, exitflag, output] = ...
                particleswarm( ...
                    cost_fun, ...
                    numVars, ...
                    lb, ...
                    ub, ...
                    options);

            runtime_seconds = ...
                toc(optimization_start);


            %% Step 14: Evaluate the optimized solution

            [final_cost, final_fidelity, final_leakage] = ...
                nested_MZM_higher_dim_cost( ...
                    x_best, ...
                    target_orders, ...
                    target_amplitudes, ...
                    target_phases, ...
                    order_range, ...
                    D);


            %% Step 15: Store the current column result

            current_column_costs(column_number) = ...
                final_cost;

            current_column_fidelities(column_number) = ...
                final_fidelity;

            current_column_leakages(column_number) = ...
                final_leakage;

            current_column_runtimes(column_number) = ...
                runtime_seconds;

            current_column_iterations(column_number) = ...
                output.iterations;

            current_column_function_evaluations(column_number) = ...
                output.funccount;

            current_column_exitflags(column_number) = ...
                exitflag;


            fprintf( ...
                ['Seed %d | U%d | Column %d/%d | ' ...
                 'Fidelity = %.6f | Leakage = %.6f | ' ...
                 'Runtime = %.2f s\n'], ...
                seed, ...
                mub_number, ...
                column_number, ...
                D, ...
                final_fidelity, ...
                final_leakage, ...
                runtime_seconds);

        end


        %% Step 16: Store the complete current-gate results

        column_fidelities_by_seed( ...
            seed_index, mub_number, :) = ...
            reshape( ...
                current_column_fidelities, ...
                1, 1, D);

        column_leakages_by_seed( ...
            seed_index, mub_number, :) = ...
            reshape( ...
                current_column_leakages, ...
                1, 1, D);

        column_costs_by_seed( ...
            seed_index, mub_number, :) = ...
            reshape( ...
                current_column_costs, ...
                1, 1, D);

        column_runtimes_by_seed( ...
            seed_index, mub_number, :) = ...
            reshape( ...
                current_column_runtimes, ...
                1, 1, D);

        column_iterations_by_seed( ...
            seed_index, mub_number, :) = ...
            reshape( ...
                current_column_iterations, ...
                1, 1, D);

        column_function_evaluations_by_seed( ...
            seed_index, mub_number, :) = ...
            reshape( ...
                current_column_function_evaluations, ...
                1, 1, D);

        column_exitflags_by_seed( ...
            seed_index, mub_number, :) = ...
            reshape( ...
                current_column_exitflags, ...
                1, 1, D);


        % Column-averaged fidelity of the complete gate.
        gate_fidelities_by_seed( ...
            seed_index, mub_number) = ...
            mean(current_column_fidelities);

        % Average leakage of the complete gate.
        gate_leakages_by_seed( ...
            seed_index, mub_number) = ...
            mean(current_column_leakages);

        % Average optimization cost of the complete gate.
        gate_costs_by_seed( ...
            seed_index, mub_number) = ...
            mean(current_column_costs);

        % Total runtime of all D columns of the gate.
        gate_runtimes_by_seed( ...
            seed_index, mub_number) = ...
            sum(current_column_runtimes);

        minimum_column_fidelities_by_seed( ...
            seed_index, mub_number) = ...
            min(current_column_fidelities);

        maximum_column_fidelities_by_seed( ...
            seed_index, mub_number) = ...
            max(current_column_fidelities);


        fprintf('\nSeed %d, U%d summary:\n', ...
            seed, ...
            mub_number);

        fprintf('Column-averaged gate fidelity: %.8f\n', ...
            gate_fidelities_by_seed( ...
                seed_index, mub_number));

        fprintf('Average gate leakage: %.8f\n', ...
            gate_leakages_by_seed( ...
                seed_index, mub_number));

        fprintf('Total gate runtime: %.2f seconds\n', ...
            gate_runtimes_by_seed( ...
                seed_index, mub_number));

    end


    %% Step 17: Display the complete current-seed summary

    seed_wall_runtimes(seed_index) = ...
        toc(seed_start);

    current_seed_average_fidelity = ...
        mean(gate_fidelities_by_seed(seed_index, :));

    current_seed_total_pso_runtime = ...
        sum(gate_runtimes_by_seed(seed_index, :));

    fprintf('\n');
    fprintf('####################################################\n');
    fprintf('SEED %d COMPLETE\n', seed);
    fprintf('Average fidelity across U1 through U%d: %.8f\n', ...
        D, ...
        current_seed_average_fidelity);

    fprintf('Total PSO runtime for this seed: %.2f seconds\n', ...
        current_seed_total_pso_runtime);

    fprintf('Complete wall-clock runtime for this seed: %.2f seconds\n', ...
        seed_wall_runtimes(seed_index));

    fprintf('####################################################\n');

end


%% Step 18: Calculate fidelity statistics across seeds

% Mean fidelity of each MUB gate.
mean_gate_fidelities = ...
    mean(gate_fidelities_by_seed, 1);

% Sample standard deviation of each MUB gate.
standard_deviation_gate_fidelities = ...
    std(gate_fidelities_by_seed, 0, 1);

% Standard error of the mean for each MUB gate.
standard_error_gate_fidelities = ...
    standard_deviation_gate_fidelities ./ sqrt(num_seeds);


%% Step 19: Calculate runtime statistics across seeds

% Mean runtime of each complete MUB gate.
mean_gate_runtimes = ...
    mean(gate_runtimes_by_seed, 1);

% Sample standard deviation of each MUB-gate runtime.
standard_deviation_gate_runtimes = ...
    std(gate_runtimes_by_seed, 0, 1);

% Standard error of the mean runtime.
standard_error_gate_runtimes = ...
    standard_deviation_gate_runtimes ./ sqrt(num_seeds);


%% Step 20: Calculate leakage and cost statistics

mean_gate_leakages = ...
    mean(gate_leakages_by_seed, 1);

standard_deviation_gate_leakages = ...
    std(gate_leakages_by_seed, 0, 1);

standard_error_gate_leakages = ...
    standard_deviation_gate_leakages ./ sqrt(num_seeds);

mean_gate_costs = ...
    mean(gate_costs_by_seed, 1);

standard_deviation_gate_costs = ...
    std(gate_costs_by_seed, 0, 1);

standard_error_gate_costs = ...
    standard_deviation_gate_costs ./ sqrt(num_seeds);


%% Step 21: Create fidelity and runtime statistics tables

mub_labels = ...
    compose('U%d', mub_numbers).';

fidelity_statistics = table( ...
    mub_labels, ...
    mean_gate_fidelities.', ...
    standard_deviation_gate_fidelities.', ...
    standard_error_gate_fidelities.', ...
    'VariableNames', { ...
        'MUB_Gate', ...
        'Mean_Fidelity', ...
        'Standard_Deviation', ...
        'Standard_Error' ...
        });

runtime_statistics = table( ...
    mub_labels, ...
    mean_gate_runtimes.', ...
    standard_deviation_gate_runtimes.', ...
    standard_error_gate_runtimes.', ...
    'VariableNames', { ...
        'MUB_Gate', ...
        'Mean_Runtime_Seconds', ...
        'Standard_Deviation_Seconds', ...
        'Standard_Error_Seconds' ...
        });


%% Step 22: Calculate complete-experiment statistics

% One average fidelity value for each seed.
overall_fidelity_by_seed = ...
    mean(gate_fidelities_by_seed, 2);

overall_mean_fidelity = ...
    mean(overall_fidelity_by_seed);

overall_standard_deviation_fidelity = ...
    std(overall_fidelity_by_seed, 0);

overall_standard_error_fidelity = ...
    overall_standard_deviation_fidelity / sqrt(num_seeds);


% One total PSO runtime value for each seed.
total_pso_runtime_by_seed = ...
    sum(gate_runtimes_by_seed, 2);

overall_mean_runtime = ...
    mean(total_pso_runtime_by_seed);

overall_standard_deviation_runtime = ...
    std(total_pso_runtime_by_seed, 0);

overall_standard_error_runtime = ...
    overall_standard_deviation_runtime / sqrt(num_seeds);


% Complete wall-clock runtime of the entire script.
complete_experiment_runtime = ...
    toc(complete_experiment_start);


%% Step 23: Display the final statistics

fprintf('\n\n');
fprintf('====================================================\n');
fprintf('FIDELITY STATISTICS ACROSS %d SEEDS\n', num_seeds);
fprintf('====================================================\n');

disp(fidelity_statistics);

fprintf('Overall mean fidelity across all MUBs and seeds: %.8f\n', ...
    overall_mean_fidelity);

fprintf('Overall fidelity standard deviation: %.8f\n', ...
    overall_standard_deviation_fidelity);

fprintf('Overall fidelity standard error: %.8f\n', ...
    overall_standard_error_fidelity);


fprintf('\n');
fprintf('====================================================\n');
fprintf('RUNTIME STATISTICS ACROSS %d SEEDS\n', num_seeds);
fprintf('====================================================\n');

disp(runtime_statistics);

fprintf('Mean total PSO runtime per seed: %.2f seconds\n', ...
    overall_mean_runtime);

fprintf('Total-runtime standard deviation: %.2f seconds\n', ...
    overall_standard_deviation_runtime);

fprintf('Total-runtime standard error: %.2f seconds\n', ...
    overall_standard_error_runtime);

fprintf('Complete seed-sweep wall-clock runtime: %.2f seconds\n', ...
    complete_experiment_runtime);

fprintf('====================================================\n');


%% Step 24: Select graph error bars

switch lower(error_bar_type)

    case 'standard deviation'

        fidelity_error_values = ...
            standard_deviation_gate_fidelities;

        runtime_error_values = ...
            standard_deviation_gate_runtimes;

        error_bar_label = ...
            'Standard deviation';

        error_bar_title = ...
            'Mean \pm 1 standard deviation';

    case 'standard error'

        fidelity_error_values = ...
            standard_error_gate_fidelities;

        runtime_error_values = ...
            standard_error_gate_runtimes;

        error_bar_label = ...
            'Standard error';

        error_bar_title = ...
            'Mean \pm 1 standard error';

    otherwise

        error( ...
            ['error_bar_type must be either ' ...
             '''standard deviation'' or ''standard error''.']);

end


%% Step 25: Plot mean fidelity across seeds

fidelity_figure = figure;

bar( ...
    mub_numbers, ...
    mean_gate_fidelities);

hold on;

errorbar( ...
    mub_numbers, ...
    mean_gate_fidelities, ...
    fidelity_error_values, ...
    'LineStyle', 'none', ...
    'LineWidth', 1.5, ...
    'CapSize', 10);

hold off;

xticks(mub_numbers);
xticklabels(compose('U%d', mub_numbers));

xlabel('MUB gate');

ylabel( ...
    sprintf( ...
        'Column-averaged fidelity with %s', ...
        lower(error_bar_label)));

title( ...
    sprintf( ...
        ['D = %d MUB Fidelity Across %d PSO Seeds\n' ...
         '%s'], ...
        D, ...
        num_seeds, ...
        error_bar_title));

grid on;
box on;

fidelity_lower_limit = ...
    max( ...
        0, ...
        min( ...
            mean_gate_fidelities - fidelity_error_values ...
            ) - 0.01);

fidelity_upper_limit = ...
    min( ...
        1, ...
        max( ...
            mean_gate_fidelities + fidelity_error_values ...
            ) + 0.01);

if fidelity_upper_limit > fidelity_lower_limit

    ylim([ ...
        fidelity_lower_limit, ...
        fidelity_upper_limit ...
        ]);

else

    ylim([0, 1]);

end


%% Step 26: Plot mean runtime across seeds

runtime_figure = figure;

bar( ...
    mub_numbers, ...
    mean_gate_runtimes);

hold on;

errorbar( ...
    mub_numbers, ...
    mean_gate_runtimes, ...
    runtime_error_values, ...
    'LineStyle', 'none', ...
    'LineWidth', 1.5, ...
    'CapSize', 10);

hold off;

xticks(mub_numbers);
xticklabels(compose('U%d', mub_numbers));

xlabel('MUB gate');

ylabel( ...
    sprintf( ...
        'Total gate runtime in seconds with %s', ...
        lower(error_bar_label)));

title( ...
    sprintf( ...
        ['D = %d MUB Runtime Across %d PSO Seeds\n' ...
         '%s'], ...
        D, ...
        num_seeds, ...
        error_bar_title));

grid on;
box on;