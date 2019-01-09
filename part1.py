from solver import *
from tchebycheff import *


if __name__ == "__main__":
    # --------------
    #  Parse data
    # --------------
    columns_to_min = ['Weight', 'Acceleration', 'Price', 'Pollution']

    # Read input data
    x = pd.read_csv('data.csv', skiprows=1,
                    dtype={'Design': np.float64, 'Frame': np.float64})

    # Fill NaN values with mean
    x.fillna(x.mean(), inplace=True)

    # Numeric columns
    columns = ['Engine', 'Torque', 'Weight', 'Acceleration',
               'Price', 'Pollution', 'Design', 'Frame']

    # min-max normalize
    x_norm = (x[columns] - x[columns].min()) / \
             (x[columns].max() - x[columns].min())

    # Convert columns to minimize
    x_norm[columns_to_min] = x_norm[columns_to_min].apply(lambda v: -v)

    # Get pareto front
    x_pareto = pareto_front(x_norm, use_cache=True)

    # ----------------------------
    #  Get default best solution
    # ----------------------------

    # Remove index column to compute ideal point and nadir point.
    x_pareto.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
    x_p = x_pareto.loc[:, x_pareto.columns != 'index']
    ipt = get_ideal_point(x_p)
    npt = get_nadir_point(x_p)

    # Compute default best solution index in Pareto set.
    best_solution_index = get_mindist_point(x_p, ipt, npt)

    # Get solution index in original data set.
    best_solution_index = int(x_pareto.iloc[best_solution_index]['index'])

    # -----------------------------
    #  Iterative decision process
    # -----------------------------

    user_satisfied = False
    while not user_satisfied:

        print("Best solution:")
        print(x.iloc[best_solution_index])

        user_input = raw_input("Are you satisfied with the solution found ? (y/n)")
        while user_input not in ('n', 'y'):
            user_input = raw_input("Please answer with 'n' or 'y' ")

        if user_input == 'y':
            user_satisfied = True
            break
        else:
            number_of_columns = len(columns)
            for i in range(number_of_columns):
                print(str(i) + ":" + str(columns[i]))

            obj = int(raw_input("Which objective would you want to improve ? (number)"))
            while obj not in range(number_of_columns):
                obj = int(raw_input("Please specify a number between 0 and "
                                    + str(number_of_columns - 1) + " "))

            keyword = "maximum" if columns[obj] in columns_to_min else "minimum"
            print("What is your " + keyword + " acceptable value for " + columns[obj] + " ?")
            value = int(raw_input())

            # Update Pareto set.
            x_pareto = reject_solution_pareto(x, x_pareto, columns,
                                              columns_to_min, obj, value)

            if x_pareto.empty:
                print("Unable to find solution with the specified preferences.")
                break

            # Remove index column to compute ideal point and nadir point.
            x_p = x_pareto.loc[:, x_pareto.columns != 'index']
            ipt = get_ideal_point(x_p)
            npt = get_nadir_point(x_p)

            # Compute default best solution index in Pareto set.
            best_solution_index = get_mindist_point(x_p, ipt, npt)

            # Get solution index in original data set.
            best_solution_index = int(x_pareto.iloc[best_solution_index]['index'])
