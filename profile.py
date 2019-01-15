import numpy as np


def create_weights(length=1, gen_type="iid"):
    weights = None

    if gen_type == "iid":  # Generate uniform weights for all objectives.
        weights = np.random.uniform(0, 100, length)

    if gen_type == "peaked":  # Generate peaked weights.
        weights = np.array([])
        nbpeaks = np.random.random_integers(1, length // 2)
        for i in range(nbpeaks):
            weights = np.append(weights, np.random.random_integers(50, 100))
        for i in range(length - nbpeaks):
            weights = np.append(weights, np.random.random_integers(0, 10))
        np.random.shuffle(weights)

    if gen_type == "ordered":  # Generate ordered weights.
        weights = np.array([])
        upper_limit = 100
        lower_limit = 0
        weights = np.append(
            weights,
            np.random.random_integers(lower_limit, upper_limit)
        )
        for i in range(length - 1):
            lower_limit = upper_limit + 1000
            upper_limit = lower_limit + 100
            weights = np.append(
                weights,
                np.random.random_integers(lower_limit, upper_limit)
            )
        np.random.shuffle(weights)

    return weights / weights.sum()


def create_weighted_sum_dm(data_set, weights=None, gen_type="iid"):
    """Computes the best solution in `data_set` in regards to a weighted sum of
    the objectives. If no argument `weights` is passed, a random weight vector
    is generated (with its values summing to one).

    Args:
        data_set (pd.DataFrame): Data matrix in which to search the best
            solution.
        weights (list): Weight vector of dimension (1, len(pareto_set)).
        gen_type (string): Type of generation for the weights. Can be 'iid',
            'peaked' or 'ordered'.

    Returns:
        (list, int) Weight vector used and index in `data_set` of the best
            solution.
    """
    if weights is None:
        weights = create_weights(length=len(data_set.columns), gen_type=gen_type)

    max_value = -float('inf')
    best_index = 0
    for i in range(len(data_set)):
        weighted_sum = (weights * data_set[data_set.columns].iloc[i]).sum()
        if weighted_sum > max_value:
            max_value = weighted_sum
            best_index = i

    return weights, best_index
