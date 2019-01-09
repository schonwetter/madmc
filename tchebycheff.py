def get_ideal_point(x_p):
    """Returns the ideal point of `X_p`, that is the point that maximises each
    component.

    Args:
        x_p (pd.DataFrame): Pareto front.

    Returns:
        (pd.Series) Ideal point.
    """
    return x_p.max()


def get_nadir_point(x_p):
    """Returns the nadir point of `X_p`, that is the point that minimises each
    component in the pareto front.

    Args:
        x_p (pd.DataFrame): Pareto front.

    Returns:
        (pd.Series) Nadir point.
    """
    return x_p.min()


def augmented_tchebycheff_dist(point, ideal_point, nadir_point, epsilon=0.001):
    """Computes the augmented tchebycheff distance of a `point` to the ideal
    point in the direction of the nadir point.

    Args:
        point (pd.Series)
        ideal_point (pd.Series)
        nadir_point (pd.Series)
        epsilon (float)

    Returns:
        (float) Augmented Tchebycheff distance.
    """
    norm_i = (ideal_point - point) / (ideal_point - nadir_point)
    max_norm = norm_i.max()
    esum = norm_i.sum() * epsilon
    return max_norm + esum


def get_mindist_point(x, ideal_point, nadir_point):
    """Computes the solution in `X` that minimises the augmented tchebycheff
    distance.

    Args:
        x (pd.DataFrame): List of solutions.
        ideal_point (pd.Series): Ideal point in X.
        nadir_point (pd.Series): Nadir point in X.

    Returns:
        (int) Index in X.
    """
    min_idx = 0
    min_dist = float('inf')
    for i in range(len(x)):
        if augmented_tchebycheff_dist(x.iloc[i], ideal_point, nadir_point) \
                < min_dist:
            min_idx = i

    return min_idx


def reject_solution_pareto(x, x_pareto, columns, columns_to_min, obj, value):
    """Filters a Pareto set.

    Args:
        x (pd.DataFrame): DataFrame from which the objective values should be
            read.
        x_pareto (pd.DataFrame): Pareto set to filter.
        columns (list): List of objectives.
        columns_to_min (list): Objectives that should be minimised.
        obj (int): Index of objective in `columns`.
        value (float or int): Right hand side of the constraint to add.

    Returns:
        (pd.DataFrame) Filtered Pareto set.
    """
    indices_to_reject = list()

    if columns[obj] in columns_to_min:
        for i in range(len(x_pareto)):
            if x[columns[obj]][x_pareto.iloc[i]['index']] > value:
                indices_to_reject.append(i)
    else:
        for i in range(len(x_pareto)):
            if x[columns[obj]][x_pareto.iloc[i]['index']] < value:
                indices_to_reject.append(i)

    x_pareto.drop(x_pareto.index[indices_to_reject], inplace=True)
    return x_pareto
