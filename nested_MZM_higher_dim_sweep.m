% Nested MZM Higher-Dimensional Dimension Sweep
% Alexander Tiu
% 7/21/2026
%
% This script:
%   1. Tests D = 4, 8, 16, and 32.
%   2. Uses one fixed PSO seed for every dimension.
%   3. Optimizes every nontrivial MUB gate U1 through UD.
%   4. Optimizes every column of each gate independently.
%   5. Calculates one column-averaged fidelity for each gate.
%   6. Calculates one total optimization runtime for each gate.
%   7. Calculates the mean, standard deviation, and standard error
%      across the gates within each dimension.
%   8. Saves one compact MAT file for each dimension.
%
% The saved standard deviation and standard error describe variation
% across MUB gates, not variation across PSO seeds.
%% ==============================================================

clc;
clear;
close all;


%% Step 1: Define the dimension-sweep experiment

% Dimensions included in the sweep.
dimensions_to_test = [4, 8, 16, 32];

% Use one fixed seed because PSO repeatability was tested separately.
seed = 1;

% Highest Bessel order retained for each RF tone.
max_bessel_order = 4;


%% Step 2: Define shared particle-swarm settings

% These settings remain fixed across dimensions so the runtime and
% fidelity results are directly comparable.
swarm_size = 100;
max_iterations = 100;
max_stall_iterations = 30;
function_tolerance = 1e-6;
inertia_range = [0.5, 1.5];


%% Step 3: Prepare file locations

% Find the folder containing this sweep script.
script_dir = fileparts(mfilename('fullpath'));

% Make sure MATLAB can find the cost function in the same folder.
addpath(script_dir);

% Create one folder for the compact dimension-sweep result files.
output_dir = fullfile( ...
    script_dir, ...
    'dimension_sweep_results');

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end


%% Step 4: Display the complete experiment plan

total_pso_calls = sum(dimensions_to_test.^2);

fprintf('====================================================\n');
fprintf('NESTED-MZM HIGHER-DIMENSIONAL SWEEP\n');
fprintf('====================================================\n');
fprintf('Dimensions: %s\n', mat2str(dimensions_to_test));
fprintf('Fixed PSO seed: %d\n', seed);
fprintf('Swarm size: %d\n', swarm_size);
fprintf('Maximum iterations: %d\n', max_iterations);
fprintf('Maximum stall iterations: %d\n', ...
    max_stall_iterations);
fprintf('Total PSO calls across all dimensions: %d\n', ...
    total_pso_calls);
fprintf('====================================================\n');


%% Step 5: Begin the dimension sweep

complete_sweep_start = tic;

for dimension_index = 1:numel(dimensions_to_test)

    D = dimensions_to_test(dimension_index);

    % Restart the random-number generator at the same seed for each
    % dimension so each dimension is independently reproducible.
    rng(seed, 'twister');

    fprintf('\n\n');
    fprintf('####################################################\n');
    fprintf('STARTING DIMENSION D = %d\n', D);
    fprintf('####################################################\n');


    %% Step 6: Calculate the nested-MZM model size

    % A D-dimensional implementation requires RF tones 1 through D - 1.
    num_tones = D - 1;

    % Each branch contains:
    %   4(D - 1) modulation depths
    %   4(D - 1) RF phases
    %   2 bias phases
    numVars = 8*num_tones + 2;

    % The nontrivial MUB gates are U1 through UD.
    mub_numbers = 1:D;

    % Every gate contains D columns.
    column_numbers = 1:D;

    fprintf('RF tones per arm: %d\n', num_tones);
    fprintf('Optimization parameters per branch: %d\n', ...
        numVars);
    fprintf('Nontrivial MUB gates: %d\n', D);
    fprintf('Columns per gate: %d\n', D);
    fprintf('PSO calls for this dimension: %d\n', D^2);


    %% Step 7: Load the D-dimensional MUB matrices

    mub_file = fullfile( ...
        script_dir, ...
        sprintf('mubs_D%d.mat', D));

    if ~isfile(mub_file)
        error( ...
            'Could not find the required MUB file: %s', ...
            mub_file);
    end

    loaded_data = load(mub_file);

    if ~isfield(loaded_data, 'U_mubs')
        error( ...
            '%s does not contain a variable named U_mubs.', ...
            mub_file);
    end

    U_mubs = loaded_data.U_mubs;

    expected_size = [D, D, D + 1];

    if ~isequal(size(U_mubs), expected_size)
        error( ...
            ['Expected U_mubs to have size %d x %d x %d, ' ...
             'but received %s.'], ...
            D, ...
            D, ...
            D + 1, ...
            mat2str(size(U_mubs)));
    end

    fprintf('Loaded MUB file: %s\n', mub_file);


    %% Step 8: Define the complete simulated frequency range

    max_frequency_order = ...
        max_bessel_order * sum(1:num_tones);

    order_range = ...
        -max_frequency_order:max_frequency_order;

    fprintf('Simulated frequency-order range: %d through %d\n', ...
        order_range(1), ...
        order_range(end));


    %% Step 9: Define the nested-MZM parameter bounds

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

    if numel(lb) ~= numVars || numel(ub) ~= numVars
        error( ...
            'The lower and upper bounds must each contain %d values.', ...
            numVars);
    end


    %% Step 10: Configure particle swarm for this dimension

    options = optimoptions( ...
        'particleswarm', ...
        'CreationFcn', @pswcreationuniform, ...
        'Display', 'off', ...
        'FunctionTolerance', function_tolerance, ...
        'InertiaRange', inertia_range, ...
        'MaxIterations', max_iterations, ...
        'MaxStallIterations', max_stall_iterations, ...
        'SwarmSize', swarm_size, ...
        'UseParallel', false, ...
        'UseVectorized', false);


    %% Step 11: Prepare compact storage for this dimension

    % One fidelity value for each complete MUB gate.
    gate_fidelities = nan(1, D);

    % One runtime value for each complete MUB gate.
    %
    % Each gate runtime is the sum of the runtimes of its D independent
    % column optimizations.
    gate_runtimes = nan(1, D);

    % Store unitarity checks for the D nontrivial MUB matrices.
    unitarity_errors = nan(1, D);

    dimension_start = tic;

    result_filename = sprintf( ...
        'nested_MZM_higher_dim_sweep_D%d.mat', ...
        D);

    result_path = fullfile( ...
        output_dir, ...
        result_filename);


    %% Step 12: Optimize every nontrivial MUB gate

    for mub_number = mub_numbers

        fprintf('\n');
        fprintf('----------------------------------------------------\n');
        fprintf('D = %d: Optimizing U%d dagger\n', ...
            D, ...
            mub_number);
        fprintf('----------------------------------------------------\n');

        % The stored basis at index 1 is U0.
        basis_index = mub_number + 1;

        % Select Um and construct the measurement gate Um dagger.
        U_m = U_mubs(:, :, basis_index);
        U_target = U_m';

        % Verify the current target matrix.
        unitarity_errors(mub_number) = ...
            norm(U_m' * U_m - eye(D), 'fro');

        if unitarity_errors(mub_number) > 1e-8
            error( ...
                'D = %d, MUB %d failed the unitarity check.', ...
                D, ...
                mub_number);
        end

        % Temporary storage for the D columns of this gate.
        current_column_fidelities = zeros(1, D);
        current_column_runtimes = zeros(1, D);


        %% Step 13: Optimize every column of the current gate

        for column_number = column_numbers

            % Extract the target column.
            target_column = ...
                U_target(:, column_number);

            target_amplitudes = ...
                abs(target_column).';

            target_phases = ...
                angle(target_column).';

            % Shift the desired output orders according to the input bin.
            target_orders = ...
                (0:D-1) - (column_number - 1);


            %% Step 14: Create the current-column objective function

            cost_fun = @(x) ...
                nested_MZM_higher_dim_cost( ...
                    x, ...
                    target_orders, ...
                    target_amplitudes, ...
                    target_phases, ...
                    order_range, ...
                    D);


            %% Step 15: Run PSO for the current column

            optimization_start = tic;

            [x_best, ~, ~, ~] = ...
                particleswarm( ...
                    cost_fun, ...
                    numVars, ...
                    lb, ...
                    ub, ...
                    options);

            runtime_seconds = ...
                toc(optimization_start);


            %% Step 16: Evaluate and store the current-column result

            [~, final_fidelity, ~] = ...
                nested_MZM_higher_dim_cost( ...
                    x_best, ...
                    target_orders, ...
                    target_amplitudes, ...
                    target_phases, ...
                    order_range, ...
                    D);

            current_column_fidelities(column_number) = ...
                final_fidelity;

            current_column_runtimes(column_number) = ...
                runtime_seconds;

            fprintf( ...
                ['D = %d | U%d | Column %d/%d | ' ...
                 'Fidelity = %.6f | Runtime = %.2f s\n'], ...
                D, ...
                mub_number, ...
                column_number, ...
                D, ...
                final_fidelity, ...
                runtime_seconds);

        end


        %% Step 17: Calculate the current complete-gate values

        % The fidelity of the current gate is the average of its D
        % independently optimized column fidelities.
        gate_fidelities(mub_number) = ...
            mean(current_column_fidelities);

        % The runtime of the current gate is the sum of its D
        % independently measured column runtimes.
        gate_runtimes(mub_number) = ...
            sum(current_column_runtimes);

        fprintf('\nD = %d, U%d summary:\n', ...
            D, ...
            mub_number);

        fprintf('Column-averaged gate fidelity: %.8f\n', ...
            gate_fidelities(mub_number));

        fprintf('Total gate runtime: %.2f seconds\n', ...
            gate_runtimes(mub_number));


        %% Step 18: Save a compact progress checkpoint

        % Save after every completed gate so a long D = 16 or D = 32
        % calculation does not lose all completed gate results.
        completed_gate_count = mub_number;
        calculation_complete = false;

        completed_gate_fidelities = ...
            gate_fidelities(1:completed_gate_count);

        completed_gate_runtimes = ...
            gate_runtimes(1:completed_gate_count);

        mean_fidelity_across_completed_gates = ...
            mean(completed_gate_fidelities);

        std_fidelity_across_completed_gates = ...
            std(completed_gate_fidelities, 0);

        se_fidelity_across_completed_gates = ...
            std_fidelity_across_completed_gates / ...
            sqrt(completed_gate_count);

        mean_runtime_across_completed_gates = ...
            mean(completed_gate_runtimes);

        std_runtime_across_completed_gates = ...
            std(completed_gate_runtimes, 0);

        se_runtime_across_completed_gates = ...
            std_runtime_across_completed_gates / ...
            sqrt(completed_gate_count);

        dimension_result = struct();

        dimension_result.D = D;
        dimension_result.seed = seed;
        dimension_result.mub_numbers = mub_numbers;
        dimension_result.completed_gate_count = ...
            completed_gate_count;
        dimension_result.calculation_complete = ...
            calculation_complete;

        dimension_result.gate_fidelities = ...
            gate_fidelities;
        dimension_result.gate_runtimes = ...
            gate_runtimes;
        dimension_result.unitarity_errors = ...
            unitarity_errors;

        dimension_result.mean_fidelity_across_gates = ...
            mean_fidelity_across_completed_gates;
        dimension_result.std_fidelity_across_gates = ...
            std_fidelity_across_completed_gates;
        dimension_result.se_fidelity_across_gates = ...
            se_fidelity_across_completed_gates;

        dimension_result.mean_runtime_across_gates = ...
            mean_runtime_across_completed_gates;
        dimension_result.std_runtime_across_gates = ...
            std_runtime_across_completed_gates;
        dimension_result.se_runtime_across_gates = ...
            se_runtime_across_completed_gates;

        dimension_result.num_tones = num_tones;
        dimension_result.numVars = numVars;
        dimension_result.num_gates = D;
        dimension_result.columns_per_gate = D;
        dimension_result.number_of_pso_calls = D^2;

        dimension_result.max_bessel_order = ...
            max_bessel_order;
        dimension_result.max_frequency_order = ...
            max_frequency_order;

        dimension_result.swarm_size = swarm_size;
        dimension_result.max_iterations = ...
            max_iterations;
        dimension_result.max_stall_iterations = ...
            max_stall_iterations;
        dimension_result.function_tolerance = ...
            function_tolerance;
        dimension_result.inertia_range = ...
            inertia_range;
        dimension_result.use_parallel = false;

        save( ...
            result_path, ...
            '-struct', ...
            'dimension_result');

        fprintf('Progress checkpoint saved to:\n%s\n', ...
            result_path);

    end


    %% Step 19: Calculate final statistics across all gates

    % These statistics describe variation among the D MUB gates.
    mean_fidelity_across_gates = ...
        mean(gate_fidelities);

    std_fidelity_across_gates = ...
        std(gate_fidelities, 0);

    se_fidelity_across_gates = ...
        std_fidelity_across_gates / sqrt(D);

    mean_runtime_across_gates = ...
        mean(gate_runtimes);

    std_runtime_across_gates = ...
        std(gate_runtimes, 0);

    se_runtime_across_gates = ...
        std_runtime_across_gates / sqrt(D);

    % Total measured PSO runtime for every gate and column in this
    % dimension.
    total_dimension_pso_runtime = ...
        sum(gate_runtimes);

    % Complete wall-clock runtime including setup, printing, and saving.
    total_dimension_wall_runtime = ...
        toc(dimension_start);


    %% Step 20: Display the final dimension summary

    fprintf('\n');
    fprintf('====================================================\n');
    fprintf('D = %d DIMENSION SWEEP COMPLETE\n', D);
    fprintf('====================================================\n');

    fprintf('Gate fidelities:\n');
    disp(gate_fidelities);

    fprintf('Gate runtimes in seconds:\n');
    disp(gate_runtimes);

    fprintf('Mean fidelity across gates: %.8f\n', ...
        mean_fidelity_across_gates);

    fprintf('Fidelity standard deviation across gates: %.8f\n', ...
        std_fidelity_across_gates);

    fprintf('Fidelity standard error across gates: %.8f\n', ...
        se_fidelity_across_gates);

    fprintf('Mean gate runtime: %.2f seconds\n', ...
        mean_runtime_across_gates);

    fprintf('Gate-runtime standard deviation: %.2f seconds\n', ...
        std_runtime_across_gates);

    fprintf('Gate-runtime standard error: %.2f seconds\n', ...
        se_runtime_across_gates);

    fprintf('Total PSO runtime for D = %d: %.2f seconds\n', ...
        D, ...
        total_dimension_pso_runtime);

    fprintf('Total wall-clock runtime for D = %d: %.2f seconds\n', ...
        D, ...
        total_dimension_wall_runtime);

    fprintf('====================================================\n');


    %% Step 21: Save the final compact dimension result

    completed_gate_count = D;
    calculation_complete = true;

    dimension_result = struct();

    dimension_result.D = D;
    dimension_result.seed = seed;
    dimension_result.mub_numbers = mub_numbers;
    dimension_result.completed_gate_count = ...
        completed_gate_count;
    dimension_result.calculation_complete = ...
        calculation_complete;

    % These arrays identify the fidelity and runtime of every gate.
    dimension_result.gate_fidelities = ...
        gate_fidelities;
    dimension_result.gate_runtimes = ...
        gate_runtimes;
    dimension_result.unitarity_errors = ...
        unitarity_errors;

    % These are the dimension-level statistics used by the plotting
    % script.
    dimension_result.mean_fidelity_across_gates = ...
        mean_fidelity_across_gates;
    dimension_result.std_fidelity_across_gates = ...
        std_fidelity_across_gates;
    dimension_result.se_fidelity_across_gates = ...
        se_fidelity_across_gates;

    dimension_result.mean_runtime_across_gates = ...
        mean_runtime_across_gates;
    dimension_result.std_runtime_across_gates = ...
        std_runtime_across_gates;
    dimension_result.se_runtime_across_gates = ...
        se_runtime_across_gates;

    dimension_result.total_dimension_pso_runtime = ...
        total_dimension_pso_runtime;
    dimension_result.total_dimension_wall_runtime = ...
        total_dimension_wall_runtime;

    % Save the model size and the exact PSO settings.
    dimension_result.num_tones = num_tones;
    dimension_result.numVars = numVars;
    dimension_result.num_gates = D;
    dimension_result.columns_per_gate = D;
    dimension_result.number_of_pso_calls = D^2;

    dimension_result.max_bessel_order = ...
        max_bessel_order;
    dimension_result.max_frequency_order = ...
        max_frequency_order;

    dimension_result.swarm_size = swarm_size;
    dimension_result.max_iterations = ...
        max_iterations;
    dimension_result.max_stall_iterations = ...
        max_stall_iterations;
    dimension_result.function_tolerance = ...
        function_tolerance;
    dimension_result.inertia_range = ...
        inertia_range;
    dimension_result.use_parallel = false;

    save( ...
        result_path, ...
        '-struct', ...
        'dimension_result');

    fprintf('\nFinal D = %d result saved to:\n%s\n', ...
        D, ...
        result_path);

end


%% Step 22: Display completion of the full sweep

complete_sweep_runtime = ...
    toc(complete_sweep_start);

fprintf('\n\n');
fprintf('####################################################\n');
fprintf('ALL DIMENSIONS COMPLETE\n');
fprintf('####################################################\n');
fprintf('Dimensions tested: %s\n', ...
    mat2str(dimensions_to_test));
fprintf('Complete sweep wall-clock runtime: %.2f seconds\n', ...
    complete_sweep_runtime);
fprintf('Result folder:\n%s\n', output_dir);
fprintf('####################################################\n');
