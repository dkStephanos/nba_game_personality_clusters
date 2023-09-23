import numpy as np
import pandas as pd
from efficient_apriori import apriori
from typing import List, Dict, Tuple

def run_apriori(dataset: pd.DataFrame, cluster: int, min_support: float = 0.5, min_confidence: float = 0.8, max_length: int = 6, save_results: bool = True) -> None:
    """
    Runs the apriori algorithm on the given dataset and saves/prints the results.

    :param dataset: DataFrame containing the dataset.
    :param cluster: Cluster number.
    :param min_support: Minimum support value for apriori.
    :param min_confidence: Minimum confidence value for apriori.
    :param max_length: Maximum length value for apriori.
    :param save_results: Boolean indicating whether to save the results to CSV.
    """
    dataset = dataset.drop(["cluster"], axis=1)
    transformed_vals = _transform_values(dataset.values)

    itemsets, rules = apriori(
        transformed_vals,
        min_support=min_support,
        min_confidence=min_confidence,
        max_length=max_length,
    )

    results = _filter_rules(rules)
    _print_results(rules, results)
    
    if save_results:
        _save_to_csv(results, dataset.columns, cluster)

def _transform_values(values: np.ndarray) -> List[List[str]]:
    indices = np.arange(values.shape[1])
    transformed = np.core.defchararray.add(
        indices.astype(str).reshape(1, -1), 
        np.where(values == False, '.0', '.1')
    )
    return transformed.tolist()


def _filter_rules(rules: List) -> Dict[str, List]:
    """
    Filters the apriori rules into wins and losses.

    :param rules: List of apriori rules.
    :return: Dictionary containing filtered rules.
    """
    results = {"wins": [], "losses": []}
    for rule in rules:
        if len(rule.rhs) == 1:
            result_type = "wins" if rule.rhs[0] == "0.1" else "losses"
            results[result_type].append(rule)
    return results

def _print_results(all_rules: List, filtered_rules: Dict[str, List]) -> None:
    """
    Prints the result summary to the console.

    :param all_rules: List of all apriori rules.
    :param filtered_rules: Dictionary containing filtered rules.
    """
    print(f"\nNumber of rules overall: {len(all_rules)}\n")
    print(f"\nNumber of rules for wins: {len(filtered_rules['wins'])}\n")
    print(f"\nNumber of rules for losses: {len(filtered_rules['losses'])}\n")

def _save_to_csv(results: Dict[str, List], columns: pd.Index, cluster: int) -> None:
    """
    Saves the processed apriori results to CSV.

    :param results: Dictionary containing filtered rules.
    :param columns: Columns of the original dataset.
    :param cluster: Cluster number.
    """
    for result_type in ["wins", "losses"]:
        processed_results = _process_apriori_results(
            sorted(results[result_type], key=lambda x: x.lift, reverse=True)[:1000],
            columns,
        )
        file_path = f'./data/apriori_results/stats.results-apriori-rules-cluster-{cluster}-{result_type}.csv'
        processed_results.to_csv(file_path)

def _process_apriori_results(results: List, columns: pd.Index) -> pd.DataFrame:
    """
    Processes the apriori results for saving to CSV.

    :param results: List of apriori rules.
    :param columns: Columns of the original dataset.
    :return: DataFrame containing the processed apriori results.
    """
    columns = ["WINorLOSS"] + list(columns)
    apriori_results = {
        "lhs": [],
        "rhs": [],
        "confidence": [],
        "lift": [],
    }
    for result in results:
        apriori_results["lhs"].append(_transform_result(result.lhs, columns))
        apriori_results["rhs"].append(_transform_result(result.rhs, columns))
        apriori_results["confidence"].append(result.confidence)
        apriori_results["lift"].append(result.lift)

    return pd.DataFrame(apriori_results)

def _transform_result(result_elements: Tuple, columns: List[str]) -> List[str]:
    """
    Transforms a result element for processing.

    :param result_elements: Tuple containing result elements.
    :param columns: Columns of the original dataset.
    :return: List of transformed result elements.
    """
    return [
        f"{columns[int(element.split('.')[0])]}-{'True' if element.split('.')[1] == '1' else 'False'}"
        for element in result_elements
    ]
