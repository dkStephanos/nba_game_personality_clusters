from apyori import apriori


def run_apriori(dataset, cluster, min_support=0.75, min_confidence=0.90, max_length=5):
    # Collecting the inferred rules in a dataframe
    dataset.drop(["cluster"], axis=1, inplace=True)

    wins = dataset.loc[dataset["WINorLOSS"] == "W"].values
    encoded_wins = []
    for index, row in enumerate(wins):
        encoded_wins.append([])
        for idx, val in enumerate(row):
            encoded_wins[index].append(
                str(idx) + ('.0' if val in [False, 'L'] else '.1')
            )

    losses = dataset.loc[dataset["WINorLOSS"] == "L"].values
    encoded_losses = []
    for index, row in enumerate(losses):
        encoded_losses.append([])
        for idx, val in enumerate(row):
            encoded_losses[index].append(
                str(idx) + ('.0' if val in [False, 'L'] else '.1')
            )

    print(len(encoded_losses), len(encoded_wins))

    association_rules = apriori(
        encoded_wins,
        min_support=min_support,
        min_confidence=min_confidence,
        max_length=max_length,
    )

    wins = []
    # Filtering our results to just rules that rhs is survived
    for result in list(association_rules):
        for entry in result.ordered_statistics:
            if entry.items_add == frozenset({'0.1'}):
                wins.append(entry)

    association_rules = apriori(
        encoded_losses,
        min_support=min_support,
        min_confidence=min_confidence,
        max_length=max_length,
    )

    losses = []
    # Filtering our results to just rules that rhs is survived
    for result in list(association_rules):
        for entry in result.ordered_statistics:
            if entry.items_add == frozenset({'0.0'}):
                losses.append(entry)

    print("\nNumber of rules for wins: {0}\n".format(len(wins)))
    print("\nNumber of rules for losses: {0}\n".format(len(losses)))

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
