from solver import *


if __name__ == "__main__":
	columns_to_min = ['Weight', 'Acceleration', 'Price', 'Pollution']

	# Read input data
	X = pd.read_csv('data.csv', skiprows=1, dtype={'Design': np.float64, 'Frame': np.float64})
	# Fill NaN values with mean
	X.fillna(X.mean(), inplace=True)
	# Numeric columns
	columns = ['Engine', 'Torque', 'Weight', 'Acceleration', 'Price', 'Pollution', 'Design', 'Frame']
	# min-max normalize
	X_norm = (X[columns] - X[columns].min()) / (X[columns].max() - X[columns].min())
	# Convert columns to minimize
	X_norm[columns_to_min] = X_norm[columns_to_min].apply(lambda x: -x)
	# Get pareto front
	X_pareto = pareto_front(X_norm, use_cache=False)

	ipt = get_ideal_point(X_pareto)
	npt = get_nadir_point(X_pareto)

	i = get_mindist_point(X_norm, ipt, npt)
	print(X.iloc[i])
