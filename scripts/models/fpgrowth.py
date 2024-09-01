import pandas as pd
from typing import Optional
from joblib import Parallel, delayed
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
from pyspark.sql import SparkSession, Row
from pyspark.sql import functions as F
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import array_contains
from pyspark import SparkConf


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
    df = df.iloc[:, 6:].astype(bool)

    # Generate frequent itemsets.
    frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)

    # Check if frequent itemsets contain 'win'.
    frequent_itemsets_win = frequent_itemsets[
        frequent_itemsets["itemsets"].apply(lambda x: "win" in x)
    ]

    if frequent_itemsets_win.empty:
        print(f"No itemsets with 'win' for min_support={min_support}")
        return  # Exit function if no relevant itemsets.

    if save_results:
        # Save results to CSV, using the cluster number in the filename.
        frequent_itemsets_win.to_csv(
            f"../data/fpgrowth_results/cluster_{cluster}_itemsets.csv", index=False
        )

    # Generate association rules.
    rules = association_rules(
        frequent_itemsets_win, metric="confidence", min_threshold=min_confidence
    )

    # Filter rules to get those with 'win'.
    rules_win = rules[rules["consequents"].apply(lambda x: "win" in x)]

    if rules_win.empty:
        print(f"No rules with 'win' for min_confidence={min_confidence}")
        return  # Exit function if no relevant rules.

    if save_results:
        # Save rules to CSV.
        rules_win.to_csv(
            f"../data/fpgrowth_results/cluster_{cluster}_rules.csv", index=False
        )

    # Optional: Print the results (can be removed in production code).
    if verbose:
        print(frequent_itemsets_win)
        print(rules_win)

def simplify_rules(pandas_df: pd.DataFrame, n_jobs: int) -> pd.DataFrame:
    def is_subset(rule_a, rule_b):
        set_a = set(rule_a.split(','))
        set_b = set(rule_b.split(','))
        return set_a.issubset(set_b)

    def simplify_group(group):
        group = group.sort_values(by=['lift', 'antecedent'], ascending=[False, True])
        selected_rules = []
        for i, row in group.iterrows():
            is_redundant = False
            for selected_rule in selected_rules:
                if is_subset(row['antecedent'], selected_rule['antecedent']):
                    is_redundant = True
                    break
            if not is_redundant:
                selected_rules.append(row)
        return selected_rules

    print("Simplifying rule sets...")
    # Use parallel processing to simplify rules
    grouped = pandas_df.groupby('consequent')
    simplified_rules = Parallel(n_jobs=n_jobs)(
        delayed(simplify_group)(group) for name, group in grouped
    )

    # Flatten the list of lists
    simplified_rules = [item for sublist in simplified_rules for item in sublist]

    return pd.DataFrame(simplified_rules)

def run_fpgrowth(
    cluster: int,
    df: pd.DataFrame,
    min_support: float = 0.2,
    min_confidence: float = 0.5,
    min_lift: float = 2.0,
    max_len: Optional[int] = None,
    verbose: bool = False,
    save_results: bool = True,
    n_jobs: int = -1,  # Number of parallel jobs (-1 means using all processors)
    num_partitions: int = 4
) -> None:
    # Preprocess DataFrame.
    df = df.iloc[:, 6:].drop(columns=['+/-_0.7', '+/-_0.8', '+/-_0.9', 'Opp.+/-_0.7', 'Opp.+/-_0.8', 'Opp.+/-_0.9']).astype(bool)

    # Create SparkConf object
    conf = SparkConf()
    conf.set("spark.executor.memory", "12g")
    conf.set("spark.driver.memory", "12g")
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

    # Initialize Spark session with the configuration
    spark = SparkSession.builder.appName("FP-growth").config(conf=conf).getOrCreate()

    # Convert truth table DataFrame to a list of transactions
    print("Converting truth table to transactions...")
    transactions = df.apply(lambda row: [col for col in row.index if row[col]], axis=1).tolist()

    # Convert transactions to PySpark Rows
    data = [Row(items=transaction) for transaction in transactions]

    # Convert Pandas DataFrame to Spark DataFrame
    sdf = spark.createDataFrame(data).repartition(num_partitions)
    
    # Generate frequent itemsets using FP-growth
    print("Fitting the fp-growth model...")
    fp_growth = FPGrowth(
        itemsCol="items", minSupport=min_support, minConfidence=min_confidence
    )
    model = fp_growth.fit(sdf)
    frequent_itemsets = model.freqItemsets

    # Check if frequent itemsets contain 'win'
    print("Evaluating itemsets...")
    condition = array_contains(frequent_itemsets["items"], "win-True") | array_contains(
        frequent_itemsets["items"], "win-False"
    )
    frequent_itemsets_win = frequent_itemsets.filter(condition)

    if frequent_itemsets_win.count() == 0:
        print(f"No itemsets with 'win' for min_support={min_support}")
        return  # Exit function if no relevant itemsets.

    if save_results:
        # Convert the array to a string, joining elements with a comma
        frequent_itemsets_win = frequent_itemsets_win.withColumn(
            "items", F.concat_ws(",", "items")
        )
        pandas_df = frequent_itemsets_win.toPandas()
        pandas_df.to_csv(
            f"../data/fpgrowth_results/cluster_{cluster}_itemsets.csv", index=False
        )

    # Generate association rules
    association_rules = model.associationRules

    # Filter rules to get those with 'win'
    print("Evaluating association rules...")
    condition = array_contains(
        association_rules["consequent"], "win-True"
    ) | array_contains(association_rules["consequent"], "win-False")
    rules_win = association_rules.filter(condition)

    if rules_win.count() == 0:
        print(f"No rules with 'win' for min_confidence={min_confidence}")
        return  # Exit function if no relevant rules.

    # Convert rules to Pandas DataFrame for simplification
    rules_win = rules_win.withColumn(
        "antecedent", F.concat_ws(",", "antecedent")
    ).withColumn("consequent", F.concat_ws(",", "consequent"))
    pandas_df = rules_win.toPandas()

    # Filter by minimum lift
    pandas_df = pandas_df[pandas_df['lift'] >= min_lift]
    print(len(pandas_df))

    # Simplify rules
    simplified_df = simplify_rules(pandas_df, n_jobs)

    if save_results:
        # Save simplified rules to CSV
        simplified_df.to_csv(
            f"../data/fpgrowth_results/cluster_{cluster}_simplified_rules.csv", index=False
        )

    # Optional: Print the results (can be removed in production code)
    if verbose:
        frequent_itemsets_win.show()
        rules_win.show()
        print(simplified_df)

    # Stop Spark session
    spark.stop()