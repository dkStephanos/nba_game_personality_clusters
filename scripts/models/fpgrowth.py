import pandas as pd
from typing import Optional
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules

def run_fpgrowth(
    cluster: int,
    df: pd.DataFrame,
    min_support: float = 0.2,
    min_confidence: float = 0.5,
    max_len: Optional[int] = None,
    verbose: bool = False,
    save_results: bool = True,
) -> None:
    # Preprocess DataFrame.
    df = df.iloc[:, 5:].astype(bool)

    # Generate frequent itemsets.
    frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)

    # Check if frequent itemsets contain 'win'.
    frequent_itemsets_win = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: 'win' in x)]

    if frequent_itemsets_win.empty:
        print(f"No itemsets with 'win' for min_support={min_support}")
        return  # Exit function if no relevant itemsets.

    if save_results:
        # Save results to CSV, using the cluster number in the filename.
        frequent_itemsets_win.to_csv(f"../data/fpgrowth_results/cluster_{cluster}_itemsets.csv", index=False)

    # Generate association rules.
    rules = association_rules(frequent_itemsets_win, metric='confidence', min_threshold=min_confidence)

    # Filter rules to get those with 'win'.
    rules_win = rules[rules['consequents'].apply(lambda x: 'win' in x)]

    if rules_win.empty:
        print(f"No rules with 'win' for min_confidence={min_confidence}")
        return  # Exit function if no relevant rules.

    if save_results:
        # Save rules to CSV.
        rules_win.to_csv(f"../data/fpgrowth_results/cluster_{cluster}_rules.csv", index=False)

    # Optional: Print the results (can be removed in production code).
    if verbose:
        print(frequent_itemsets_win)
        print(rules_win)
