import enum
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

    wins = []
    losses = []
    # Filtering our results to just rules that rhs is survived
    for result in list(association_rules):
        for entry in result.ordered_statistics:
            if entry.items_add == frozenset({'0.1'}):
                wins.append(entry)
            elif entry.items_add == frozenset({'0.0'}):
                losses.append(entry)

    print("\nNumber of rules for wins: {0}\n".format(len(wins)))
    print("\nNumber of rules for losses: {0}\n".format(len(losses)))

    if save_results:
        # Sorting by lift
        sortedResults = sorted(wins, key=lambda x: x.lift, reverse=True)
        with open(
            f'./data/stats.results-apriori-rules-cluster-{cluster}-wins.txt', 'w+'
        ) as f:
            for result in sortedResults[:1000]:
                f.write(str(result))
        sortedResults = sorted(losses, key=lambda x: x.lift, reverse=True)
        with open(
            f'./data/stats.results-apriori-rules-cluster-{cluster}-losses.txt', 'w+'
        ) as f:
            for result in sortedResults[:1000]:
                f.write(str(result))


def process_apriori_results(results):
    apriori_results = []
    results = results.split('OrderedStatistic')
    for idx, result in enumerate(results[1:]):
        parsed_result = {}

        # splitting base result
        results[idx] = result.split("=")

        # Stripping off items_base
        for val in results[idx][1].split(", items_add")[0][11:-2].split(", "):
            if "items_base" not in parsed_result:
                parsed_result["items_base"] = []
            parsed_result["items_base"].append(val[1:-1])

        # Stripping off items_add
        for val in results[idx][2].split(", confidence")[0][11:-2].split(", "):
            if "items_add" not in parsed_result:
                parsed_result["items_add"] = []
            parsed_result["items_add"].append(val[1:-1])

        # Stripping off lift
        parsed_result["confidence"] = float(results[idx][3].split(', ')[0])

        # Stripping off lift
        parsed_result["lift"] = float(results[idx][4][:-1])

        # Append the parsed result to the full set
        apriori_results.append(parsed_result)
    print(apriori_results)
