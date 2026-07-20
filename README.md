# qce
Contains runtime and error analysis for MUBs

mubs_benchmark checks the runtimes for N number of runs for MUB generation. 

nested_MZM_higher_dim_cost.m

How function works:
    - Determine the number of RF tones
    - Unpack the parameter vectors: four sets of modulation depths, four sets of RF phases, two bias phases
    - Calculate the spectrum from each of the four arms
    - Combine the arms using the two bias phases
    - Build the ideal target frequency-bin vector
    - Compare the simulated output with the target using fidelity
    - Return the cost function: cost = 1 - F

Local helpers:

calculate_arm_spectrum
    - Calculate frequency-bin output from one arm driven by all D-1 RF tones
    - Simplifies the three nested loops required for D=4

nested_MZM_higher_dim_opt.m

How function works:
    - Controls the experiment and particle swarm optimization
    - Block 1: Experiment settings. Define D=8, which MUB is being tested, seeds, running one column or full gate
    - Block 2: Load target MUB. Construct Um dagger
    - Block 3: Define hardware model
    - Block 4: Define the frequency-bin range
    - Block 5: Configure PSO
    - Block 6: Prepare one target column
    - Block 7: Call cost function
    - Block 8: Record results
    - Block 9: Expand only after pilot works
    