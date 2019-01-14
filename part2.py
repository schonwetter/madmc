import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gurobipy import *

from profile import create_weighted_sum_dm
from tchebycheff import get_ideal_nadir


def pairwise_max_regret(x, y, known_preferences):
    """Computes the Pairwise Max Regret (PMR) of alternative `x` w.r.t. `y`.

    Args:
        x (pd.Series): Alternative x.
        y (pd.Series): Alternative y.
        known_preferences: Set of pairs `p` such that p[0] is known to be
            preferred to p[1].

    Returns:
        PMR value and weights at optimum.
    """
    m = Model("PMR")
    weights_var = []

    # Create variables.
    for i in range(len(columns)):
        weights_var.append(m.addVar(vtype=GRB.CONTINUOUS,
                                    name="x" + str(i), lb=0, ub=1))
        m.update()

    # Create objective.
    obj = quicksum([weights_var[i] * y[columns[i]] - weights_var[i] * x[columns[i]]
                    for i in range(len(columns))])
    m.setObjective(obj, GRB.MAXIMIZE)

    # Create constraints.
    m.addConstr(quicksum(weights_var) == 1)

    for pair in known_preferences:
        yy = pair[0]
        xx = pair[1]
        m.addConstr(quicksum([weights_var[i] * (yy[columns[i]] - xx[columns[i]])
                              for i in range(len(columns))]) >= 0)

    # Solve and retrieve value.
    m.setParam('OutputFlag', False)
    m.optimize()
    weights_values = [variable.x for variable in weights_var]
    return m.objVal, weights_values


def max_regret(x, data_set, known_preferences):
    """Computes the maximum regret (MR) of an alternative `x` in the data set
    `data_set`.

    Args:
        x (pd.Series): Alternative x.
        data_set (pd.DataFrame): Data set.
        known_preferences: Set of pairs `p` such that p[0] is known to be
            preferred to p[1].

    Returns:
        Max regret value,
        index of y alternative corresponding to the regret,
        weight values used.
    """
    max_value = -float('inf')
    max_y = -1
    current_w = []
    for y_idx in range(len(data_set)):
        value, weights = pairwise_max_regret(x, data_set.iloc[y_idx],
                                             known_preferences)
        if value > max_value:
            max_value = value
            max_y = y_idx
            current_w = weights
    return max_value, max_y, current_w


def minimax_regret(data_set, known_preferences):
    """Computes the minimax regret (MMR) over the data set specified.

    Args:
        data_set (pd.DataFrame): Data set.
        known_preferences: Set of pairs `p` such that p[0] is known to be
            preferred to p[1].

    Returns:
        index of alternative x that minimises max regret,
        index of alternative y that maximises regret,
        weight values used,
        value of minimax regret.
    """
    min_value = float('inf')
    best_x = -1
    best_y = -1
    current_w = []

    for x in range(len(data_set)):
        value, current_y, weight = max_regret(data_set.iloc[x],
                                              data_set, known_preferences)
        if value < min_value:
            min_value = value
            best_x = x
            best_y = current_y
            current_w = weight

    return best_x, best_y, current_w, min_value


def fitness(weight, x):
    """Returns the sum of the values in `x` weighted by the values in `weights`.
    """
    return (weight * x[columns]).sum()


def get_best_solution(weights, data_set):
    """Computes the best solution in the data set with respect to the weighted
    sum of the solution's values.

    Args:
        weights (list): Weights to use.
        data_set (pd.DataFrame): Data set in which to search the solution.

    Returns:
        (int) Index of the best solution in `data_set`.
    """
    index = -1
    max_value = -float('inf')
    for i in range(len(data_set)):
        weighted_sum = fitness(weights, data_set.iloc[i])
        if weighted_sum > max_value:
            max_value = weighted_sum
            index = i

    return index


def current_solution_strategy(data_set, display=False, gen_type="iid"):
    """Starts the Current Solution Strategy (CSS). As long as the Decision Maker
    (DM) is not satisfied, the iterative process will ask them to specify their
    preference between the two alternatives:
    - x that minimises MR.
    - y that maximises PMR for x.

    Args:
        data_set (pd.DataFrame): Normalized data set.
        display (bool): If set to True, the function will display a plot of the
            evolution of MMR during the procedure.
        gen_type (string): Type of weight generation used to generate the DM
            profile. See `create_weighted_sum_dm`.

    Returns:
        (int) Number of questions asked before the DM was satisfied.
    """

    # Create a preference profile for the decision maker.
    # `_weights_dm` are the weights attributed to each objective by the DM.
    # `_index_dm` is the index of the best solution in the data set according to
    # the weights.
    _weights_dm, _index_dm = create_weighted_sum_dm(data_set, gen_type=gen_type)
    index_css = -1
    known_preferences = []

    if display:
        x_ticks = []
        mmr_values = []

    cpt_question = 0
    while _index_dm != index_css:
        # Compute MMR.
        x, y, weights_css, mmr_val = minimax_regret(data_set, known_preferences)
        index_css = x

        if display:
            x_ticks.append(cpt_question)
            mmr_values.append(mmr_val)

        print("{:*^50}".format("Iteration {}".format(cpt_question)))
        print("MMR = {:.4f}".format(mmr_val))
        print("Index of preferred car: {}".format(_index_dm))
        print("Index of MMR: {}".format(index_css))

        if _index_dm == index_css:
            print("Decision maker satisfied.")
            break

        print("Next question: car No {} > car No {} ?".format(x, y))

        # Ask DM's preference between x and y.
        f_x = fitness(_weights_dm, data_set.iloc[x])
        f_y = fitness(_weights_dm, data_set.iloc[y])
        if f_x >= f_y:
            known_preferences.append((data_set.iloc[x], data_set.iloc[y]))
        else:
            known_preferences.append((data_set.iloc[y], data_set.iloc[x]))

        cpt_question += 1

    if display:
        plt.plot(x_ticks, mmr_values, 'bo')
        plt.plot(x_ticks, mmr_values)
        plt.ylabel('MMR')
        plt.xlabel('Number of questions')
        plt.title('Evolution of MMR depending on the number of queries')
        plt.show()
    return cpt_question


def average_number_queries_required(data_set, nbiter=20, gen_type="idd"):
    """Computes the average number of question needed to find DM's favorite
    solution.

    Args:
        data_set (pd.DataFrame): Normalized data set.
        nbiter (int): Number of iteration used.
        gen_type (string): Type of weight generation used to generate the DM
            profile. See `create_weighted_sum_dm`.

    Returns:
        (float) Average Number of questions asked.
    """
    cpt_questions = 0
    for i in range(nbiter):
        cpt_questions += current_solution_strategy(data_set,
                                                   display=False,
                                                   gen_type=gen_type)
    return float(cpt_questions) / nbiter


if __name__ == "__main__":
    # --------------
    #  Parse data
    # --------------
    columns_to_min = ['Weight', 'Acceleration', 'Price', 'Pollution']

    # Read input data
    car_data_set = pd.read_csv('data.csv', skiprows=1,
                           dtype={'Design': np.float64, 'Frame': np.float64})

    # Numeric columns
    columns = ['Engine', 'Torque', 'Weight', 'Acceleration', 'Price', 'Pollution']

    # min-max normalize
    x_norm = car_data_set.loc[:, columns]
    ipt, npt = get_ideal_nadir(x_norm)
    x_norm = x_norm / (ipt - npt)

    # Convert columns to minimize
    x_norm[columns_to_min] = x_norm[columns_to_min].apply(lambda v: -v)

    # Try different weight initialisation:
    # - iid
    weights_dm, _ = create_weighted_sum_dm(x_norm, gen_type="iid")
    print(weights_dm)
    # - peaked
    weights_dm, _ = create_weighted_sum_dm(x_norm, gen_type="peaked")
    print(weights_dm)
    # - ordered
    weights_dm, _ = create_weighted_sum_dm(x_norm, gen_type="ordered")
    print(weights_dm)

    # Start CSS
    current_solution_strategy(x_norm, display=True, gen_type="ordered")

    #Average number of questions required to find DM's favorite car
    # with nbiter=100 , av_num=1.28 (with uniform iid distribution of weight).
    # with nbiter=100 , av_num= 2.67 (with uniform peaked distribution of weight).
    # with nbiter=100 , av_num=2.29 (with uniform ordered distribution of weight).
    #av_numb=average_number_querries_required(x_norm,nbiter=100,param="peaked")
    #print("Average number of querries : "+str(av_numb))
    #av_numb=average_number_querries_required(x_norm,nbiter=100,param="ordered")
    #print("Average number of querries : "+str(av_numb))

