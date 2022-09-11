from apyori import apriori

def run_apriori(dataset, min_support=0.005, min_confidence=0.8, min_length=2):
    # Collecting the inferred rules in a dataframe 
    association_rules = list(apriori(dataset.values, min_support=min_support, min_confidence=min_confidence, min_length=min_length))

    wins = []
    losses = []
    #Filtering our results to just rules that rhs is survived
    for result in association_rules:
        for entry in result.ordered_statistics:
            if entry.items_add == frozenset({'W'}):
                wins.append(entry)
            else:
                losses.append(entry)

    print("\nNumber of rules for wins: {0}\n".format(len(wins)))
    print("\nNumber of rules for losses: {0}\n".format(len(losses)))

    #Sorting by lift
    for unsortedResults in [wins, losses]:
        sortedResults = sorted(unsortedResults, key=lambda x: x.lift, reverse=True)
        for result in sortedResults:
            print(str(result))
    