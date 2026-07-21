%% ==============================================================
% Nested MZM Higher Dimensional Cost Function
% Alexander Tiu 7/20/2026
%% ==============================================================

function [cost, fidelity, leakage_fraction] = ...
    nested_MZM_higher_dim_cost( ...
    x, ...
    target_orders, ...
    target_amplitudes, ...
    target_phases, ...
    order_range, ...
    D)
    fidelity = 0;
    leakage_fraction = 1;

    %% Step 1: Validate and organize inputs

    % Dimension must be a positive integer greater than one.
    if ~isscalar(D) || D < 2 || D ~= floor(D)
        error('D must be an integer greater than or equal to 2.');
    end

    % RF drive contains one tone for each harmonic.
    num_tones = D - 1;
    expected_num_vars = 8 * num_tones + 2;

    % Convert the supplied arrays into row vectors.
    x = x(:).';
    target_orders = target_orders(:).';
    target_amplitudes = target_amplitudes(:).';
    target_phases = target_phases(:).';
    order_range = order_range(:).';

    % Check the number of parameters.
    if numel(x) ~= expected_num_vars
        error( ...
            'Expected %d parameters for D = %d, but received %d.', ...
            expected_num_vars, D, numel(x));
    end

    % Each target order must have one amplitude and one phase.
    if numel(target_orders) ~= numel(target_amplitudes) || ...
            numel(target_orders) ~= numel(target_phases)

        error([ ...
            'target_orders, target_amplitudes, and target_phases ' ...
            'must have the same number of elements.']);
    end

    % Every target order must exist within order_range.
    if ~all(ismember(target_orders, order_range))
        error('Every target order must be included in order_range.');
    end


    %% Step 2: Unpack nested-MZM parameters

    % Indices of the four groups of modulation depths.
    beta_start = 1;
    beta_end = 4 * num_tones;

    % Indices of the four groups of RF phases.
    phi_start = beta_end + 1;
    phi_end = beta_end + 4 * num_tones;

    % beta has size 4 x (D - 1).
    beta = reshape( ...
        x(beta_start:beta_end), ...
        [num_tones, 4]).';

    % phi also has size 4 x (D - 1).
    phi = reshape( ...
        x(phi_start:phi_end), ...
        [num_tones, 4]).';

    % Final two parameters are bias phases.
    phi_bias_1 = x(phi_end + 1);
    phi_bias_2 = x(phi_end + 2);


    %% Step 3: Calculate the spectrum for each modulated arm

    % Match the Bessel truncation used in the original D = 4 model.
    max_bessel_order = 4;

    E_arm_1 = calculate_arm_spectrum( ...
        beta(1, :), ...
        phi(1, :), ...
        order_range, ...
        max_bessel_order);

    E_arm_2 = calculate_arm_spectrum( ...
        beta(2, :), ...
        phi(2, :), ...
        order_range, ...
        max_bessel_order);

    E_arm_3 = calculate_arm_spectrum( ...
        beta(3, :), ...
        phi(3, :), ...
        order_range, ...
        max_bessel_order);

    E_arm_4 = calculate_arm_spectrum( ...
        beta(4, :), ...
        phi(4, :), ...
        order_range, ...
        max_bessel_order);


    %% Step 4: Combine the four arms

    % First internal MZI pair.
    E_I = 0.5 * ( ...
        E_arm_1 + ...
        E_arm_2 .* exp(1j * phi_bias_1));

    % Second internal MZI pair.
    E_Q = 0.5 * ( ...
        E_arm_3 + ...
        E_arm_4 .* exp(1j * phi_bias_2));

    % Single modeled nested-MZM output.
    E_circuit = (E_I + 1j * E_Q) / sqrt(2);


    %% Step 5: Build the ideal target spectrum

    % The target is defined over the complete simulated order range.
    psi_target = zeros(1, numel(order_range));

    % Target orders must not be repeated.
    if numel(unique(target_orders)) ~= numel(target_orders)
        error('target_orders must not contain duplicate frequency orders.');
    end

    % Insert each desired complex target coefficient.
    for target_idx = 1:numel(target_orders)

        order_index = find( ...
            order_range == target_orders(target_idx), ...
            1, ...
            'first');

        psi_target(order_index) = ...
            target_amplitudes(target_idx) .* ...
            exp(1j * target_phases(target_idx));
    end

    % Desired computational orders.
    target_mask = ismember(order_range, target_orders);

    % Undesired sideband orders.
    out_of_band_mask = ~target_mask;


    %% Step 6: Calculate in-band mismatch and out-of-band leakage

    % Total optical power across all simulated bins.
    total_power = sum(abs(E_circuit).^2);

    % Reject a candidate that produces essentially no output.
    if total_power <= eps
        cost = 1e6;
        return;
    end

    % Extract the desired frequency-bin space.
    E_in_band = E_circuit(target_mask);
    psi_target_in_band = psi_target(target_mask);

    circuit_in_band_norm = norm(E_in_band);
    target_norm = norm(psi_target_in_band);

    % Target must contain at least one nonzero coefficient.
    if target_norm <= eps
        error('The target spectrum must contain at least one nonzero amplitude.');
    end

    % Reject a candidate with no output in the desired bins.
    if circuit_in_band_norm <= eps
        cost = 1e6;
        return;
    end

    % Normalize the realized and target fields.
    psi_circuit_normalized = ...
        E_in_band / circuit_in_band_norm;

    psi_target_normalized = ...
        psi_target_in_band / target_norm;

    % State fidelity.
    fidelity = abs( ...
        psi_target_normalized * psi_circuit_normalized' ...
        )^2;

    % In-band mismatch.
    cost_in_band = 1 - fidelity;

    % Fraction of power outside the desired frequency bins.
    out_of_band_power = ...
        sum(abs(E_circuit(out_of_band_mask)).^2);

    leakage_fraction = out_of_band_power / total_power;

    % Final scalar objective minimized by PSO.
    cost = cost_in_band + leakage_fraction;

end


%% Calculate the spectrum produced by one phase-modulator arm

function E = calculate_arm_spectrum( ...
    beta_arm, ...
    phi_arm, ...
    order_range, ...
    max_bessel_order)

    %% Step 1: Validate and organize inputs

    beta_arm = beta_arm(:).';
    phi_arm = phi_arm(:).';
    order_range = order_range(:).';

    if numel(beta_arm) ~= numel(phi_arm)
        error('beta_arm and phi_arm must have the same length.');
    end

    if ~isscalar(max_bessel_order) || ...
            max_bessel_order < 0 || ...
            max_bessel_order ~= floor(max_bessel_order)

        error('max_bessel_order must be a nonnegative integer.');
    end


    %% Step 2: Begin with an unmodulated optical field

    % Initially, all amplitude is located at frequency order zero.
    spectrum = 1;
    current_max_order = 0;


    %% Step 3: Add each RF tone to the spectrum

    num_tones = numel(beta_arm);
    q_values = -max_bessel_order:max_bessel_order;

    for tone = 1:num_tones

        % Physical frequency shifts generated by this RF harmonic.
        tone_shifts = tone * q_values;

        % Complex Bessel coefficients.
        tone_coefficients = ...
            besselj(q_values, beta_arm(tone)) .* ...
            exp(1j * q_values * phi_arm(tone));

        % Largest order generated by this individual tone.
        tone_max_order = tone * max_bessel_order;

        % Sparse spectrum for the current tone.
        tone_spectrum = zeros(1, 2 * tone_max_order + 1);

        % Convert physical orders to MATLAB indices.
        tone_indices = tone_shifts + tone_max_order + 1;

        % Insert the tone coefficients.
        tone_spectrum(tone_indices) = tone_coefficients;

        % Combine this tone with the preceding tones.
        spectrum = conv(spectrum, tone_spectrum);

        % Update the represented frequency-order range.
        current_max_order = current_max_order + tone_max_order;
    end


    %% Step 4: Extract the requested frequency-bin orders

    E = zeros(1, numel(order_range));

    for idx = 1:numel(order_range)

        physical_order = order_range(idx);

        spectrum_index = ...
            physical_order + current_max_order + 1;

        if spectrum_index >= 1 && ...
                spectrum_index <= numel(spectrum)

            E(idx) = spectrum(spectrum_index);
        end
    end

end