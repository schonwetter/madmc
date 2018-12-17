import pandas as pd
import numpy as np
import math


def _compute_pareto_front(X):
	"""Computes the pareto front from a list of vectors to maximise.

	Args:
		X (pd.DataFrame): Input data.

	Returns:
		(pd.DataFrame) Pareto front of the input data.
	"""
	X_pareto = pd.DataFrame([])
	dominates = {}
	for i in range(len(X)):
		for j in range(len(X)):
			if i == j: continue
			if np.all(X.iloc[i] >= X.iloc[j]) and np.any(X.iloc[i] > X.iloc[j]):
				dominates[(i, j)] = True

	for i in range(len(X)):
		if not any([dominates.get((j, i), False) for j in range(len(X))]):
			X_pareto = X_pareto.append(X.iloc[i])

	return X_pareto


def pareto_front(X, use_cache=True):
	"""Computes the pareto front from a list of vectors to maximise. Uses
	`pareto.csv` as a cache. If the file exists, its content is returned.

	Args:
		X (pd.DataFrame): Input data.
		use_cache (bool): If false, force computation of pareto front and
			saves it in `pareto.csv`.

	Returns:
		(pd.DataFrame) Pareto front of the input data.
	"""
	if use_cache:
		try:
			return pd.read_csv('pareto.csv', usecols=lambda name: name != "")
		except IOError:
			print("No pareto file. Computing pareto front.")

	X_pareto = _compute_pareto_front(X)
	X_pareto.to_csv(path_or_buf='pareto.csv')
	return X_pareto


def get_ideal_point(X_p):
	"""Returns the ideal point of `X_p`, that is the point that maximises each component

	Args:
		X_p (pd.DataFrame): Pareto front.

	Returns: (pd.Series) Ideal point.
	"""
	return X_p.max()


def get_nadir_point(X_p):
	"""Returns the nadir point of `X_p`, that is the point that minimises each component
	among the pareto front.

	Args:
		X_p (pd.DataFrame): Pareto front.

	Returns: (pd.Series) Nadir point.
	"""
	return X_p.min()


def augmented_tchebycheff_dist(point, ideal_point, nadir_point, epsilon=0.001):
	"""Computes the augmented tchebycheff distance of a `point` to the ideal point
	in the direction of the nadir point.

	Args:
		point (pd.Series)
		ideal_point (pd.Series)
		nadir_point (pd.Series)
		epsilon (float)

	Returns: (float) Augmented Tchebycheff distance.
	"""
	norm_i = (ideal_point - point) / (ideal_point - nadir_point)
	max_norm = norm_i.max()
	esum = norm_i.sum() * epsilon
	return max_norm + esum


def get_mindist_point(X, ideal_point, nadir_point):
	"""
	"""
	min_idx = 0
	min_dist = math.inf
	for i in range(len(X)):
		if augmented_tchebycheff_dist(X.iloc[i], ideal_point, nadir_point) < min_dist:
			min_idx = i

	return min_idx