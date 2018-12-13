import math
from copy import deepcopy


def augmented_tchebycheff_norm(point, reference, epsilon=0.001):
	max_distance = 0
	for category_idx in range(1, len(point)):
		dist = reference[category_idx-1] -


def get_ideal_point(X, directions):
	ideal_point = [0 if dir == 'max' else math.inf for dir in directions]
	for line in X:
		for category_idx in range(len(ideal_point)):
			if directions[category_idx] == 'max' and line[category_idx+1] > ideal_point[category_idx]:
				ideal_point[category_idx] = line[category_idx+1]
			if directions[category_idx] == 'min' and line[category_idx+1] < ideal_point[category_idx]:
				ideal_point[category_idx] = line[category_idx+1]
	return ideal_point


def mean_wo_nan(x):
	x = [i for i in x if not math.isnan(i)]
	return sum(x) / len(x)


def parse_data(filename):
	X = list()
	i = 0
	design_nan = list()
	frame_nan = list()

	with open(filename, 'r') as f:
		labels = f.readline()
		data = f.read().split("\n")
		for line in data:
			linedata = line.strip().split(',')
			X.append([linedata[0]] + list(map(float, linedata[1:])))

			if math.isnan(X[i][-1]): frame_nan.append(i)
			if math.isnan(X[i][-2]): design_nan.append(i)
			i += 1

	mean_frame = mean_wo_nan([row[-1] for row in X])
	mean_design = mean_wo_nan([row[-2] for row in X])
	for i in range(len(X)):
		if i in frame_nan: X[i][-1] = mean_frame
		if i in design_nan: X[i][-2] = mean_design

	return X, labels


def normalize(X):
	X_copy = deepcopy(X)
	for category in range(1, len(X_copy[0])):
		values = list(line[category] for line in X_copy)
		max_value = max(values)
		min_value = min(values)
		for line in X_copy:
			line[category] = (line[category] - min_value) / (max_value - min_value)
	return X_copy

if __name__ == "__main__":
	X, labels = parse_data('data.csv')
	directions = ['max', 'max', 'min', 'min', 'min', 'min', 'max', 'max']
	X_normalized = normalize(X)
	print(get_ideal_point(X_normalized, directions))