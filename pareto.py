import pandas as pd
import numpy as np


def _compute_pareto_front(x):
	"""Computes the pareto front from a list of vectors to maximise.

	Args:
		x (pd.DataFrame): Input data.

	Returns:
		(pd.DataFrame) Pareto front of the input data.
	"""
	x_pareto = pd.DataFrame([])
	dominates = {}
	for i in range(len(x)):
		for j in range(len(x)):
			if i == j:
				continue
			if np.all(x.iloc[i] >= x.iloc[j]) and np.any(x.iloc[i] > x.iloc[j]):
				dominates[(i, j)] = True

	for i in range(len(x)):
		if not any([dominates.get((j, i), False) for j in range(len(x))]):
			x_pareto = x_pareto.append(x.iloc[i])

	return x_pareto


def pareto_front(x, use_cache=True):
	"""Computes the pareto front from a list of vectors to maximise. Uses
	`pareto.csv` as a cache. If the file exists, its content is returned.

	Args:
		x (pd.DataFrame): Input data.
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

	x_pareto = _compute_pareto_front(x)
	x_pareto.to_csv(path_or_buf='pareto.csv')

	return x_pareto
