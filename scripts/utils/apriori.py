import pandas as pd
from apyori import apriori


def run_apriori(
    dataset,
    cluster,
    min_support=0.05,
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

    association_rules = apriori(
        temp_vals,
        min_support=min_support,
        min_confidence=min_confidence,
        max_length=max_length,
    )

    results = {"wins": [], "losses": []}
    # Filtering our results to just rules that rhs is survived
    for result in list(association_rules):
        for entry in result.ordered_statistics:
            if entry.items_add == frozenset({'0.1'}):
                results["wins"].append(entry)
            elif entry.items_add == frozenset({'0.0'}):
                results["losses"].append(entry)

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
    apriori_results = {
        "items_base": [],
        "items_add": [],
        "confidence": [],
        "lift": [],
    }
    results = results.split('OrderedStatistic')
    for idx, result in enumerate(results[1:]):
        # splitting base result
        results[idx] = result.split("=")

        # Stripping off items_base
        tmp_list = []
        for val in results[idx][1].split(", items_add")[0][11:-2].split(", "):
            tmp_str = val[1:-1].split('.')
            tmp_list.append(
                f"{columns[int(tmp_str[0])]}-{'True' if tmp_str[1] == '1' else 'False'}"
            )
        apriori_results["items_base"].append(tmp_list)

        # Stripping off items_add
        tmp_list = []
        for val in results[idx][2].split(", confidence")[0][11:-2].split(", "):
            tmp_str = val[1:-1].split('.')
            tmp_list.append(
                f"{columns[int(tmp_str[0])]}-{'True' if tmp_str[1] == '1' else 'False'}"
            )
        apriori_results["items_add"].append(tmp_list)

        # Stripping off lift
        apriori_results["confidence"].append(float(results[idx][3].split(', ')[0]))

        # Stripping off lift
        apriori_results["lift"].append(float(results[idx][4][:-1]))

    return pd.DataFrame(apriori_results)
