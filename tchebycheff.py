from operator import lt, le

import pandas as pd


def get_ideal_point(x_p):
    """Returns the ideal point of `x_p`, that is the point that maximises each
    component.

    Args:
        x_p (pd.DataFrame): Pareto front.

    Returns:
        (pd.Series) Ideal point.
    """
    return x_p.max()


def get_nadir_point(x_p):
    """Returns the nadir point of `x_p`, that is the point that minimises each
    component in the pareto front.

    Args:
        x_p (pd.DataFrame): Pareto front.

    Returns:
        (pd.Series) Nadir point.
    """
    return x_p.min()


def get_ideal_nadir(x):
    """Returns the ideal point and an approximation of the nadir point in x.

    Args:
        x (pd.DataFrame): Data set.

    Returns:
        (pd.Series) Ideal point,
        (pd.Series) Nadir point approximation.
    """
    ideal_point = dict()
    ideals_idx = list()
    nadir_point = dict()

    # Compute ideal point.
    for col_idx in x.columns:
        idx = x[col_idx].idxmax()
        ideal_point[col_idx] = x.loc[idx, col_idx]
        ideals_idx.append(idx)

    # Approximate nadir point.
    for col_idx in x.columns:
        nadir_point[col_idx] = x.loc[ideals_idx, col_idx].min()

    return pd.Series(ideal_point), pd.Series(nadir_point)


def augmented_tchebycheff_dist(point, ideal_point, nadir_point, epsilon=0.001):
    """Computes the augmented tchebycheff distance of a `point` to the ideal
    point in the direction of the nadir point. Uniform weights are used to
    control objective importance.

    Args:
        point (pd.Series)
        ideal_point (pd.Series)
        nadir_point (pd.Series)
        epsilon (float)

    Returns:
        (float) Augmented Tchebycheff distance.
    """
    if ideal_point.equals(nadir_point):
        return 0

    norm_i = (ideal_point - point) / (ideal_point - nadir_point)
    max_norm = norm_i.max()
    e_sum = norm_i.sum() * epsilon
    return max_norm + e_sum


def get_mindist_point(x, ideal_point, nadir_point):
    """Computes the solution in `x` that minimises the augmented tchebycheff
    distance.

    Args:
        x (pd.DataFrame): List of solutions.
        ideal_point (pd.Series): Ideal point in `x`.
        nadir_point (pd.Series): Nadir point in `x`.

    Returns:
        (int) Index in `x`.
    """
    min_idx = -1
    min_dist = float('inf')
    for i in x.index:
        distance = augmented_tchebycheff_dist(x.loc[i, :], ideal_point, nadir_point)
        if distance < min_dist:
            min_idx = i
            min_dist = distance

    return min_idx


def reject_solution(x_to_filter, objective_name, value, strict=False):
    """Filters a data set.

    Args:
        x_to_filter (pd.DataFrame): Data set to filter.
        objective_name (string): Name of the objective to filter.
        value (float or int): Right hand side of the constraint to add.
        strict (bool): Remove solutions when the objective value is equal to
            the `value` passed.

    Returns:
        (pd.DataFrame) Filtered data set.
    """
    indices_to_reject = list()
    op = le if strict else lt

    for i in x_to_filter.index:
        if op(x_to_filter.loc[i, objective_name], value):
            indices_to_reject.append(i)

    x_to_filter.drop(indices_to_reject, inplace=True)
    return x_to_filter
