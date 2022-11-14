import pandas as pd
from efficient_apriori import apriori


def run_apriori(
    dataset,
    cluster,
    min_support=0.5,
    min_confidence=0.8,
    max_length=6,
    save_results=True,
):
    # Collecting the inferred rules in a dataframe
    dataset.drop(["cluster"], axis=1, inplace=True)
    vals = dataset.values
    temp_vals = []
    for index, row in enumerate(vals):
        temp_vals.append([])
        for idx, val in enumerate(row):
            temp_vals[index].append(str(idx) + ('.0' if val in [False, 'L'] else '.1'))

    itemsets, rules = apriori(
        temp_vals,
        min_support=min_support,
        min_confidence=min_confidence,
        max_length=max_length,
    )

    results = {"wins": [], "losses": []}
    # Filtering our results to just rules that rhs is survived
    for rule in rules:
        if len(rule.rhs) == 1:
            if rule.rhs[0] == "0.1":
                results["wins"].append(rule)
            elif rule.rhs[0] == "0.0":
                results["losses"].append(rule)

    print("\nNumber of rules overall: {0}\n".format(len(rules)))
    print("\nNumber of rules for wins: {0}\n".format(len(results["wins"])))
    print("\nNumber of rules for losses: {0}\n".format(len(results["losses"])))

    if save_results:
        # Sorting by lift
        for result in ["wins", "losses"]:
            processed_results = process_apriori_results(
                sorted(results[result], key=lambda x: x.lift, reverse=True)[:1000],
                list(dataset.columns),
            )
            processed_results.to_csv(
                f'./data/apriori_results/stats.results-apriori-rules-cluster-{cluster}-{result}.csv'
            )


def process_apriori_results(results, columns):
    columns = ["WINorLOSS"] + columns
    apriori_results = {
        "lhs": [],
        "rhs": [],
        "confidence": [],
        "lift": [],
    }
    for result in results:
        # Stripping off lhs
        tmp_list = []
        for element in list(result.lhs):
            tmp_element = element.split('.')
            tmp_list.append(
                f"{columns[int(tmp_element[0])]}-{'True' if tmp_element[1] == '1' else 'False'}"
            )
        apriori_results["lhs"].append(tmp_list)

        # Stripping off rhs
        tmp_list = []
        for element in list(result.rhs):
            tmp_element = element.split('.')
            tmp_list.append(
                f"{columns[int(tmp_element[0])]}-{'True' if tmp_element[1] == '1' else 'False'}"
            )
        apriori_results["rhs"].append(tmp_list)

        apriori_results["confidence"].append(result.confidence)
        apriori_results["lift"].append(result.lift)

    return pd.DataFrame(apriori_results)
