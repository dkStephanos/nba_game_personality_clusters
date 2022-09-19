from apyori import apriori


def run_apriori(dataset, cluster, min_support=0.01, min_confidence=0.9, max_length=4):
    # Collecting the inferred rules in a dataframe
    dataset.drop(["cluster"], axis=1, inplace=True)
    vals = dataset.values
    temp_vals = []
    for index, row in enumerate(vals[:10]):
        temp_vals.append([])
        for idx, val in enumerate(row):
            temp_vals[index].append(dataset.columns[idx] + str(val))

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
            print(entry)
            if entry.items_add == frozenset({'WINorLOSSW'}):
                wins.append(entry)
            elif entry.items_add == frozenset({'WINorLOSSL'}):
                losses.append(entry)

    print("\nNumber of rules for wins: {0}\n".format(len(wins)))
    print("\nNumber of rules for losses: {0}\n".format(len(losses)))

    # Sorting by lift
    sortedResults = sorted(wins, key=lambda x: x.lift, reverse=True)
    with open(f'./data/stats.results-apriori-rules-cluster-{cluster}-wins.txt') as f:
        for result in sortedResults[:1000]:
            f.write(str(result))
    sortedResults = sorted(losses, key=lambda x: x.lift, reverse=True)
    with open(f'./data/stats.results-apriori-rules-cluster-{cluster}-losses.txt') as f:
        for result in sortedResults[:1000]:
            f.write(str(result))
