"""
Alexander Tiu 07/16/2026

Analyze the runtimes for each D dimension MUB
"""

import time
import matplotlib.pyplot as plt
import numpy as np
from mub_gen import mubs, verify

SEED = 0

dimensions = [
    4, 5, 7, 8, 9,
    11, 13, 16, 17, 19,
    23, 25, 27, 29, 31,
    32, 37, 41, 43, 47, 49
]

# Need a function, for the dimensions D in dimensions, execute the MUB generator and retrieve
# runtimes
def runtime_MUBs(dimensions, SEED):

    # hold the runtimes
    runtimes = []

    for D in dimensions:

        # Runtime block
        start = time.perf_counter()

        bases, info = mubs(D, seed=SEED)

        end = time.perf_counter()
        
        runtime = end - start 

        # append each runtime to runtimes
        runtimes.append(runtime)

        # Get the bases info using verify
        good, worst = verify(bases, D)

        # print results if you need to
        # print("=" * 40)
        # print(f"Dimension: {D}")
        # print(f"Runtime: {runtime:.8f} seconds")
        # print(f"MUBs generated: {info['count']}")
        # print(f"Verified: {good}")
        # print(f"Worst deviation: {worst:.2e}")

    return runtimes

def runtimes_trials(dimensions, SEED, COUNTS=5):

    results = {D: [] for D in dimensions}

    # Repeat arbitrary amount of times
    for _ in range(COUNTS):

        runtimes = runtime_MUBs(dimensions, SEED)

        # Store each runtime under its matching dimension key
        # zip pairs values in matching positions
        for D, runtime in zip(dimensions, runtimes):
            results[D].append(runtime)

    return results

def statistics(results):

    # get the standard deviation and mean of each trial in the keys
    stats = {}

    for dimension, runtimes in results.items():
        # calculate the mean stdev and se for each runtime
        mean = np.mean(runtimes)
        if len(runtimes) > 1:
            stdev = np.std(runtimes, ddof=1)
            se = stdev / np.sqrt(len(runtimes))
        else:
            stdev = 0.0
            se = 0.0

        # dimension is key, runtimes are the runtimes per key, in a list
        stats[dimension] = {
            "mean": mean,
            "stdev": stdev,
            "se": se
        }

    return stats

# Standard MUB Graph. Only runtimes versus dimensions
def standard_graph(runtimes, dimensions):

    plt.plot(dimensions, runtimes, marker="o", markersize=7)

    plt.xlabel("Dimension D")
    plt.ylabel("Runtime (seconds)")
    plt.title("MUB Generator Runtime vs Dimension")

    plt.show()


def graph_trials(stats, dimensions):

    means = [stats[D]["mean"] for D in dimensions]
    standard_errors = [stats[D]["se"] for D in dimensions]

    plt.errorbar(
        dimensions,
        means,
        yerr=standard_errors,
        fmt="o-",
        markersize=7,
        capsize=5
    )

    plt.xlabel("Dimension D")
    plt.ylabel("Mean Runtime (seconds)")
    plt.title("Mean MUB Runtime vs Dimension with Standard Error")

    plt.show()


def main():
    # Get the runtimes for each MUB, only one trial
    print("[status] Test run for D dimensions, one trial")
    runtimes = runtime_MUBs(dimensions, SEED)

    # Plot the runtimes
    standard_graph(runtimes, dimensions)

    # Number of trials
    count = 100
    print(f"[status] Test run complete. Running {count} trials")

    results = runtimes_trials(dimensions, SEED, COUNTS=count)

    # Extracting statistics
    print(f"[status] Extracting statistics")

    # in the form of dimensions: {mean: , stdev: , se: }
    stats = statistics(results)

    graph_trials(stats, dimensions)

    # Print the stats
    # for dimension, dimension_stats in stats.items():
    #     print(f"{dimension}: ")

    #     for stat_name, stat_value in dimension_stats.items():
    #         print(f"{stat_name}: {stat_value}")
    

if __name__ == "__main__":
    main()





#  mub_gen.py D # summary: count, maximal possible, verified?
# python mub_gen.py D 