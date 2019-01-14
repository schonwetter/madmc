import sys

from profile import create_weighted_sum_dm
from solver import *
from tchebycheff import *


def interactive_decision(_t_data_set):
    """Finds the best solution for the user according to the augmented
    tchebycheff distance. As long as the user is not satisfied, the data set is
    filtered and new solutions are suggested to the Decision Maker.

    Args:
        _t_data_set (pd.DataFrame): Data set of alternatives.
    """

    # Compute ideal point and nadir point (approx).
    ipt, npt = get_ideal_nadir(_t_data_set)

    # Get index of default best solution
    _best_solution_index = get_mindist_point(_t_data_set, ipt, npt)

    while True:

        print("Best solution:")
        print(data_set.loc[_best_solution_index, :])

        user_input = raw_input("Are you satisfied with the solution found ? (y/n)")
        while user_input not in ('n', 'y'):
            user_input = raw_input("Please answer with 'n' or 'y' ")

        if user_input == 'y':
            break
        else:
            number_of_columns = len(columns)
            for i in range(number_of_columns):
                print(str(i) + ":" + str(columns[i]))

            obj = int(raw_input("Which objective would you want to improve ? (number)"))
            while obj not in range(number_of_columns):
                obj = int(raw_input("Please specify a number between 0 and "
                                    + str(number_of_columns - 1) + " "))

            objective_name = columns[obj]
            keyword = "maximum" if objective_name in columns_to_min else "minimum"
            print("What is your " + keyword + " acceptable value for " + objective_name + " ?")
            value = int(raw_input())
            if objective_name in columns_to_min:
                value = - value

            # Update data set.
            _t_data_set = reject_solution(_t_data_set, objective_name, value)

            if _t_data_set.empty:
                print("Unable to find solution with the specified preferences.")
                break

            # Remove index column to compute ideal point and nadir point.
            ipt, npt = get_ideal_nadir(_t_data_set)

            # Compute best solution index in resulting data set.
            _best_solution_index = get_mindist_point(_t_data_set, ipt, npt)


def auto_iterative_decision(_t_data_set):
    """Finds the best solution for a simulated Decision Maker (DM). Its profile
    is generated as a set of random weights and its ideal solution is computed
    as the alternative that maximises the weighted sum of the objectives.

    This procedure searches for this solution in the data set by iteratively
    calculating the minimum tchebycheff distance and filtering the data set by
    the objective that's the furthest from the ideal solution of the DM.

    Args:
        _t_data_set (pd.DataFrame): Data set of alternatives.
    """

    # Best solution for a random DM profile.
    _, index_dm = create_weighted_sum_dm(_t_data_set, gen_type='peaked')
    best_solution_dm = _t_data_set.loc[index_dm, :]

    niter = 0
    while True:
        niter += 1

        # Compute ideal point and nadir point (approx).
        ipt, npt = get_ideal_nadir(_t_data_set)

        # Compute best solution in current data set.
        _best_solution_index = get_mindist_point(_t_data_set, ipt, npt)

        if _best_solution_index == index_dm:  # DM satisfied.
            break

        # Search for the objective that's the least satisfied.
        worst_objective = None
        worst_gap = 0
        for objective_name in _t_data_set.columns:
            gap = best_solution_dm[objective_name] - _t_data_set.loc[_best_solution_index, objective_name]
            gap /= (ipt[objective_name] - npt[objective_name])

            if gap > worst_gap:
                worst_gap = gap
                worst_objective = objective_name

        # Filter data set for the worst objective.
        objective_value = _t_data_set.loc[_best_solution_index, worst_objective]
        _t_data_set = reject_solution(_t_data_set, worst_objective, objective_value, strict=True)

    print("Found solution for DM in {} iterations".format(niter))


if __name__ == "__main__":
    # --------------
    #  Parse data
    # --------------
    columns_to_min = ['Weight', 'Acceleration', 'Price', 'Pollution']

    # Read input data
    data_set = pd.read_csv('data.csv', skiprows=1,
                           dtype={'Design': np.float64, 'Frame': np.float64})

    # Numeric columns
    columns = ['Engine', 'Torque', 'Weight', 'Acceleration',
               'Price', 'Pollution']

    # Working data set
    t_data_set = data_set.loc[:, columns]

    # Convert columns to minimize
    t_data_set[columns_to_min] = t_data_set[columns_to_min].apply(lambda v: -v)

    # -----------------------------
    #  Iterative decision process
    # -----------------------------

    if len(sys.argv) > 1 and sys.argv[1] == '-i':
        interactive_decision(t_data_set)
    else:
        auto_iterative_decision(t_data_set)
