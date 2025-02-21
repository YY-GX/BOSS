import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Full dataset
data = {
    "BC-Trans": {
        "ori": [
            [0.95, 0.8333333333, 0.8166666667, 0.6333333333, 0.9666666667, 0.9166666667, 0.4166666667],
            [0.95, 0.8333333333, 0.8166666667, 0.6333333333, 0.9666666667, 0.9166666667, 0.4166666667],
            [0.9833333333, 0.9, 0.9166666667, 0.7333333333, 0.7333333333],
            [0.9833333333, 0.9, 0.9166666667, 0.7333333333, 0.7333333333],
            [1, 1, 0.2666666667, 0.9833333333, 0.4666666667],
            [1, 1, 0.2666666667, 0.9833333333, 0.4666666667],
        ],
        "modified": [
            [0.9, 1, 0.5, 0, 0.65, 0.45, 0.8],
            [0.3, 0.9, 0, 0.65, 1, 0.25, 0.25],
            [0.35, 1, 0.8, 0.45, 0.9],
            [0.9, 0.75, 0.8, 0.85, 0.55],
            [1, 1, 1, 0.5, 0.15],
            [0.15, 0.95, 0.85, 0.6, 0.8],
        ],
    },
    "BC-RNN": {
        "ori": [
            [0.6333333333, 0.05, 0.15, 0, 0.06666666667, 0.6333333333, 0],
            [0.6333333333, 0.05, 0.15, 0, 0.06666666667, 0.6333333333, 0],
            [0.9833333333, 0, 0.2666666667, 0.25, 0.6333333333],
            [0.9833333333, 0, 0.2666666667, 0.25, 0.6333333333],
            [0.5166666667, 0.8666666667, 0, 0.03333333333, 0.15],
            [0.5166666667, 0.8666666667, 0, 0.03333333333, 0.15],
        ],
        "modified": [
            [0.5, 0, 0, 0, 0, 0.15, 0],
            [0, 0.2, 0, 0.5, 0.8, 0, 0],
            [0.35, 1, 0.05, 0, 0.25],
            [0.25, 0.05, 0.55, 0.6, 0.55],
            [0.8, 0.35, 0.65, 0.55, 0],
            [0, 0, 0.05, 0.7, 0.35],
        ],
    },
    "BC-Vilt": {
        "ori": [
            [0.9833333333, 0.8666666667, 0.6833333333, 0.55, 0.9833333333, 0.9, 0.9166666667],
            [0.9833333333, 0.8666666667, 0.6833333333, 0.55, 0.9833333333, 0.9, 0.9166666667],
            [1, 0.9833333333, 0.9833333333, 0.7, 0.6333333333],
            [1, 0.9833333333, 0.9833333333, 0.7, 0.6333333333],
            [1, 0.9833333333, 0.5, 0.9333333333, 0.4833333333],
            [1, 0.9833333333, 0.5, 0.9333333333, 0.4833333333],
        ],
        "modified": [
            [0.95, 1, 0.9, 0, 0.25, 0.6, 0.75],
            [0.75, 1, 0.95, 0.65, 0.95, 0.95, 0.6],
            [0.35, 1, 0.8, 0.45, 1],
            [0.95, 0.75, 0.8, 0.85, 0.55],
            [1, 1, 1, 0.8, 0.15],
            [0.25, 0.95, 0.85, 0.6, 0.8],
        ],
    },
}

# Variable to choose inclusion of zero-drop cases
INCLUDE_ZERO_DROP = True

def analyze_relationship(algorithm_name, ori_data, mod_data):
    all_spearman_corr = []
    all_p_values = []

    for i, (ori, mod) in enumerate(zip(ori_data, mod_data)):
        ori = np.array(ori)
        mod = np.array(mod)

        # Remove cases where original success rate is zero
        non_zero_indices = ori != 0
        ori = ori[non_zero_indices]
        mod = mod[non_zero_indices]

        # Calculate Relative Drop
        delta_sr = ori - mod
        delta_sr[delta_sr < 0] = 0  # Treat surprising increases as zero delta
        relative_drop = delta_sr / ori

        # Filter out zero-drop cases if chosen
        if not INCLUDE_ZERO_DROP:
            non_zero_drop_indices = relative_drop > 0
            ori = ori[non_zero_drop_indices]
            relative_drop = relative_drop[non_zero_drop_indices]

        # Debugging Outputs
        print(f"\n--- Pair {i+1} ({algorithm_name}) ---")
        print(f"Original Success Rates (filtered): {ori}")
        print(f"Modified Success Rates (filtered): {mod}")
        print(f"Relative Drop: {relative_drop}")

        # Skip empty or constant arrays
        if len(ori) == 0 or len(relative_drop) == 0 or np.all(ori == ori[0]) or np.all(relative_drop == relative_drop[0]):
            print(f"Skipping pair {i+1} due to constant or empty input.")
            continue

        # Spearman Correlation
        spearman_corr, spearman_pval = spearmanr(ori, relative_drop)
        print(f"spearman_corr, spearman_pval: {spearman_corr, spearman_pval}")

        # Save results
        all_spearman_corr.append(spearman_corr)
        all_p_values.append(spearman_pval)

        # Visualization for each pair
        plt.scatter(ori, relative_drop, label=f'Pair {i+1} (Spearman: {spearman_corr:.2f})')
        plt.xlabel('Original Success Rate')
        plt.ylabel('Relative Drop')
        plt.title(f'Pairwise Relationship for {algorithm_name}')
        plt.legend()
        plt.show()

    # Aggregate results across all pairs
    if len(all_spearman_corr) > 0:
        avg_corr = np.mean(all_spearman_corr)
        avg_pval = np.mean(all_p_values)
    else:
        avg_corr = np.nan
        avg_pval = np.nan

    # Print results
    print(f"\nResults for {algorithm_name}:")
    print(f"  Average Spearman correlation: {avg_corr:.2f}")
    print(f"  Average p-value: {avg_pval:.3f}")
    print("\nConclusion:")
    if not np.isnan(avg_pval) and avg_pval < 0.05:
        if avg_corr < 0:
            print("  There exists a significant **increasing monotonic relationship** (relative drop decreases as success rate increases).")
        else:
            print("  There exists a significant **decreasing monotonic relationship** (relative drop increases as success rate increases).")
    else:
        print("  No significant monotonic relationship.")


# Run analysis for each algorithm
for algorithm, data_dict in data.items():
    analyze_relationship(algorithm, data_dict["ori"], data_dict["modified"])
