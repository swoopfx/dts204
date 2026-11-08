"""Search algorithms implementation and performance comparison for DTS304.

Contains:
- Binary Search for single & all occurrences in a DataFrame
- Sequential (Linear) Search on unindexed DataFrame
- Simulation and benchmarking suite
"""

import time
import pandas as pd


# =======================================================================
# 1. Binary Search Functions (Requires Sorted Data)
# =======================================================================
def binary_search_product(df, target_product_id, column="product_ID"):
    """Performs a binary search for a specific product ID in a pandas DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        target_product_id (str): The product ID to search for (e.g.,
        'ELEC-1001').
        column (str): The column containing product IDs (default: 'product_ID').

    Returns:
        pd.Series or None: The matching row if found, otherwise None.
    """
    sorted_df = df.sort_values(by=column).reset_index(drop=True)

    low = 0
    high = len(sorted_df) - 1

    while low <= high:
        mid = (low + high) // 2
        mid_val = sorted_df.iloc[mid][column]

        if mid_val == target_product_id:
            return sorted_df.iloc[mid]
        elif mid_val < target_product_id:
            low = mid + 1
        else:
            high = mid - 1

    return None


def binary_search_all_products(df, target_product_id, column="product_ID"):
    """Performs binary search and returns all rows matching the target product ID,

    handling cases where a product appears across multiple purchases.
    """
    sorted_df = df.sort_values(by=column).reset_index(drop=True)
    low, high = 0, len(sorted_df) - 1
    found_idx = -1

    while low <= high:
        mid = (low + high) // 2
        mid_val = sorted_df.iloc[mid][column]

        if mid_val == target_product_id:
            found_idx = mid
            break
        elif mid_val < target_product_id:
            low = mid + 1
        else:
            high = mid - 1

    if found_idx == -1:
        return pd.DataFrame()

    start = found_idx
    while start > 0 and sorted_df.iloc[start - 1][column] == target_product_id:
        start -= 1
    end = found_idx
    while (
        end < len(sorted_df) - 1
        and sorted_df.iloc[end + 1][column] == target_product_id
    ):
        end += 1

    return sorted_df.iloc[start : end + 1]


# =======================================================================
# 2. Sequential Search Function (Unindexed Data)
# =======================================================================
def sequential_search_product(df, target_product_id, column="product_ID"):
    """Performs a sequential (linear) search for a specific product ID in an

    unsorted, unindexed DataFrame.

    Parameters:
        df (pd.DataFrame): The unindexed DataFrame.
        target_product_id (str): The product ID to search for.
        column (str): The column containing product IDs (default: 'product_ID').

    Returns:
        pd.Series or None: The matching row if found, otherwise None.
    """
    for i in range(len(df)):
        if df.iloc[i][column] == target_product_id:
            return df.iloc[i]
    return None


# =======================================================================
# 3. Performance Simulation & Comparison Function
# =======================================================================
def simulate_search_performance(
    df, targets, iterations=500, column="product_ID"
):
    """Simulates and compares performance between Sequential Search (unindexed)

    and Binary Search (sorted index).
    """
    sorted_df = df.sort_values(by=column).reset_index(drop=True)
    results = []

    for target in targets:
        # Count comparisons for Sequential Search
        seq_comps = 0
        for i in range(len(df)):
            seq_comps += 1
            if df.iloc[i][column] == target:
                break

        # Count comparisons for Binary Search
        bin_comps = 0
        low, high = 0, len(sorted_df) - 1
        while low <= high:
            bin_comps += 1
            mid = (low + high) // 2
            val = sorted_df.iloc[mid][column]
            if val == target:
                break
            elif val < target:
                low = mid + 1
            else:
                high = mid - 1

        # Benchmark timing for Sequential Search
        t0 = time.perf_counter()
        for _ in range(iterations):
            sequential_search_product(df, target, column)
        seq_time_us = ((time.perf_counter() - t0) / iterations) * 1_000_000

        # Benchmark timing for Binary Search
        t0 = time.perf_counter()
        for _ in range(iterations):
            binary_search_product(sorted_df, target, column)
        bin_time_us = ((time.perf_counter() - t0) / iterations) * 1_000_000

        speedup = seq_time_us / bin_time_us if bin_time_us > 0 else float("inf")

        if seq_comps == 1:
            scenario = "Best Case (Row 0)"
        elif target not in df[column].values:
            scenario = "Worst Case (Not Found)"
        else:
            scenario = "Typical / Average Case"

        results.append(
            {
                "Target ID": target,
                "Scenario": scenario,
                "Seq Comps": seq_comps,
                "Bin Comps": bin_comps,
                "Seq Time (µs)": round(seq_time_us, 2),
                "Bin Time (µs)": round(bin_time_us, 2),
                "Speedup (Seq/Bin)": f"{speedup:.2f}x",
            }
        )

    return pd.DataFrame(results)


if __name__ == "__main__":
    csv_file = "purchases_product.csv"
    print(f"Loading {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} records.\n")

    # 1. Binary Search Single Match
    target = "ELEC-1001"
    print(f"--- 1. Binary Search for '{target}' ---")
    res = binary_search_product(df, target)
    print(res)

    # 2. Sequential Search Single Match
    print(f"\n--- 2. Sequential Search for '{target}' ---")
    res_seq = sequential_search_product(df, target)
    print(res_seq)

    # 3. Performance Simulation
    print("\n--- 3. Performance Simulation Benchmark ---")
    test_cases = ["ELEC-1001", "HOME-5004", "NON-EXISTENT"]
    comparison = simulate_search_performance(df, test_cases, iterations=300)
    print(comparison.to_string(index=False))

