import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules


def run_fpgrowth(min_support=0.2, min_confidence=0.5):
    # Assuming df is your truth table DataFrame
    # Ensure that df is a binary matrix (0 and 1 values)

    df = pd.read_csv(
        "../data/cluster_results/cluster.stats.results-truth-table.csv", index_col=0
    )
    df = df[df["cluster"] == 0]
    df = df.iloc[:, 5:]
    df = df.drop(columns=["cluster"])
    df = df.astype(bool)

    # Applying the FP-growth algorithm to find frequent itemsets
    frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)

    # Filtering frequent itemsets to only include those containing 'win' being true
    frequent_itemsets_win_true = frequent_itemsets[
        frequent_itemsets["itemsets"].apply(lambda x: "win" in x)
    ]

    if len(frequent_itemsets_win_true) > 0:
        # Displaying the filtered results
        print(frequent_itemsets_win_true)

        # Generating association rules from the filtered frequent itemsets
        rules = association_rules(
            frequent_itemsets_win_true,
            metric="confidence",
            min_threshold=min_confidence,
            support_only=True,
        )

        # Further filtering rules to only include those where 'win' is true in the consequents
        rules_win_true = rules[
            rules["consequents"].apply(
                lambda x: "win" in x and list(df["win"][x.index]) == [1] * len(x)
            )
        ]

        if len(rules_win_true) > 0:
            print(rules_win_true)
        else:
            print(f"No rules generated for wins with min_confidence={min_confidence}")
    else:
        print(f"No itemsets generated for wins with min_support={min_support}")
