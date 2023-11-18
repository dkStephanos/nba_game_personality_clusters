import pandas as pd
from typing import Optional
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
from pyspark.sql import SparkSession, Row
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import array, array_contains, concat, lit


def mlxtend_fpgrowth(
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
    
    # Initialize Spark session
    spark = SparkSession.builder.appName("FP-growth").getOrCreate()

    # Convert truth table DataFrame to a list of transactions
    transactions = df.apply(lambda row: [f"{item}-{row[item]}" for item in row.index if row[item]], axis=1).tolist()

    # Convert transactions to PySpark Rows
    data = [Row(items=transaction) for transaction in transactions]
        
    # Convert Pandas DataFrame to Spark DataFrame
    sdf = spark.createDataFrame(data)
    
    # Generate frequent itemsets using FP-growth
    fp_growth = FPGrowth(itemsCol="items", minSupport=min_support, minConfidence=min_confidence)
    model = fp_growth.fit(sdf)
    frequent_itemsets = model.freqItemsets
    
    print(frequent_itemsets)
    
    # Check if frequent itemsets contain 'win'
    frequent_itemsets_win = frequent_itemsets.filter(array_contains(frequent_itemsets.items, 'win'))
    
    if frequent_itemsets_win.count() == 0:
        print(f"No itemsets with 'win' for min_support={min_support}")
        return  # Exit function if no relevant itemsets.
    
    if save_results:
        # Save results to CSV, using the cluster number in the filename
        frequent_itemsets_win.write.csv(f"../data/fpgrowth_results/cluster_{cluster}_itemsets.csv", header=True)
    
    # Generate association rules
    association_rules = model.associationRules
    
    print(association_rules)
    
    # Filter rules to get those with 'win'
    rules_win = association_rules.filter(association_rules.consequent.contains('win'))
    
    if rules_win.count() == 0:
        print(f"No rules with 'win' for min_confidence={min_confidence}")
        return  # Exit function if no relevant rules.
    
    if save_results:
        # Save rules to CSV
        rules_win.write.csv(f"../data/fpgrowth_results/cluster_{cluster}_rules.csv", header=True)
    
    # Optional: Print the results (can be removed in production code)
    if verbose:
        frequent_itemsets_win.show()
        rules_win.show()

    # Stop Spark session
    spark.stop()
