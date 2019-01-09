from solver import *
from tchebycheff import *


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

    # ----------------------------
    #  Get default best solution
    # ----------------------------

    # Compute ideal point and nadir point.
    ipt, npt = get_ideal_nadir(t_data_set)

    # Get index of default best solution
    best_solution_index = get_mindist_point(t_data_set, ipt, npt)

    # -----------------------------
    #  Iterative decision process
    # -----------------------------

    user_satisfied = False
    while not user_satisfied:

        print("Best solution:")
        print(data_set.loc[best_solution_index, :])

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

            objective_name = columns[obj]
            keyword = "maximum" if objective_name in columns_to_min else "minimum"
            print("What is your " + keyword + " acceptable value for " + objective_name + " ?")
            value = int(raw_input())
            if objective_name in columns_to_min:
                value = - value

            # Update data set.
            t_data_set = reject_solution(t_data_set, objective_name, value)

            if t_data_set.empty:
                print("Unable to find solution with the specified preferences.")
                break

            # Remove index column to compute ideal point and nadir point.
            ipt, npt = get_ideal_nadir(t_data_set)

            # Compute best solution index in resulting data set.
            best_solution_index = get_mindist_point(t_data_set, ipt, npt)
