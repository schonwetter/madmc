from gurobipy import *
from random import randint
from profile import create_weights


def generate_instance(n=5, p=10):
    """Generates a random multi-objective knapsack problem. Weights are random
    integers between 1 and 10. Utilities are random integers between 1 and 10.
    Bound is defined as half of the total weight.

    Args:
        n (int): Number of objectives.
        p (int): Number of objects.

    Returns:
        (list[int]): Weights,
        (list[list[int]]): Utilities for each objective and each object,
        (int): KP Bound
    """
    item_weights = list()
    item_utilities = list()
    for item in range(p):
        item_utilities.append(list())
        for _ in range(n):
            item_utilities[item].append(randint(1, 10))

        item_weights.append(randint(1, 10))

    _bound = sum(item_weights) / 2

    return item_weights, item_utilities, _bound


def get_ideal_nadir(_weights, _utilities, _bound, constraints=None):
    """Returns the ideal point and a nadir point approximation for a knapsack
    (KP) defined by `_weights`, `_utilities`, and `_bound`.

    Args:
        _weights (list[int]): List of weights for each object of the problem.
        _utilities (list[list[int]]): Utilities for each object and each objective.
        _bound (int): KP bound.
        constraints (dict): Additional constraints for (KP).

    Returns:
        (list[float], list[float], list[float]): Ideal and nadir point approximation.
    """
    n = len(_utilities[0])
    p = len(_weights)

    ideal_point = list()
    ideals = list()
    ideals_d = list()
    nadir_point = [float('inf') for _ in range(n)]
    if constraints is None:
        constraints = dict()

    # Performs `n` mono-objective optimisations.
    for _criterion in range(n):
        model = Model("mono-Knapsack")
        _variables = []

        for obj in range(p):
            _variables.append(
                model.addVar(vtype=GRB.INTEGER, lb=0, ub=1, name="x_{}".format(obj))
            )
        model.update()

        # Knapsack constraint
        model.addConstr(quicksum(_variables[i] * _weights[i] for i in range(p)) <= _bound)

        # Additional constraints
        for c, c_constant in constraints.items():
            model.addConstr(
                quicksum(_variables[i] * _utilities[i][c] for i in range(p))
                >= c_constant
            )

        objective = quicksum(_variables[i] * _utilities[i][_criterion] for i in range(p))
        model.setObjective(objective, GRB.MAXIMIZE)
        model.setParam('OutputFlag', False)
        model.optimize()
        ideals.append([sum(_variables[i].x * _utilities[i][c] for i in range(p)) for c in range(n)])
        ideals_d.append([abs(_variables[i].x) for i in range(p)])
        ideal_point.append(model.objVal)

    # Compute nadir point with the vectors computed above.
    for ideal in ideals:
        for _criterion in range(n):
            if ideal[_criterion] < nadir_point[_criterion]:
                nadir_point[_criterion] = ideal[_criterion]

    return ideal_point, nadir_point, ideals_d[0]


def get_model(_weights, _utilities, _b, epsilon=0.001, constraints=None):
    """Creates a Gurobi MOKP model that minimises the augmented Tchebycheff
    distance.

    Args:
        _weights (list[int]): Weights for each object.
        _utilities (list[list[int]]): Utilities for each object and each
            objective.
        _b (int): KP bound.
        epsilon (float): Epsilon used to augment the tchebycheff distance.
        constraints (dict): Additional constraints.

    Returns:
        (Model, list[Var], list[float], list[float], list[float]): Model,
            variables created, ideal point and nadir point, ideal point in
            decision space.
    """
    n = len(_utilities[0])
    p = len(_weights)

    if constraints is None:
        constraints = dict()
    _ipt, _npt, _ipt_d = get_ideal_nadir(_weights, _utilities, _b,
                                         constraints=constraints)

    model = Model("multi-Knapsack")
    _variables = list()
    for obj in range(p):
        _variables.append(
            model.addVar(vtype=GRB.INTEGER, lb=0, ub=1, name="x_{}".format(obj))
        )

    z = model.addVar(vtype=GRB.CONTINUOUS, name="z")
    model.update()

    # Knapsack constraint
    model.addConstr(quicksum(_variables[i] * _weights[i] for i in range(p)) <= _b)

    # Additional constraints
    for _c, c_constant in constraints.items():
        model.addConstr(
            quicksum(_variables[i] * _utilities[i][_c] for i in range(p))
            >= c_constant
        )

    augmented_sum_m = list()
    for _criterion in range(n):
        if _ipt[_criterion] == _npt[_criterion]:
            continue
        rhs = _ipt[_criterion] - quicksum(_variables[i] * _utilities[i][_criterion] for i in range(p))
        rhs /= (_ipt[_criterion] - _npt[_criterion])
        augmented_sum_m.append(rhs)
        model.addConstr(z >= rhs)

    model.setObjective(z + epsilon * quicksum(augmented_sum_m), GRB.MINIMIZE)
    model.setParam('OutputFlag', False)
    return model, _variables, _ipt, _npt, _ipt_d


def get_best_solution_dm(_objective_weights, _weights, _utilities, _b):
    """Computes the optimal solution of a MOKP as a weighted sum of the objectives.

    Args:
        _objective_weights (list[float]): Weights used for each objective.
        _weights (list[int]): Weights for each object.
        _utilities (list[list[int]]): Utilities for each object and each
            objective.
        _b (int): KP Bound.

    Returns:
        (list[float]): Optimal solution.
    """
    n = len(_utilities[0])
    p = len(_weights)
    model = Model("weighted-sum-Knapsack")
    _vars = list()
    for obj in range(p):
        _vars.append(
            model.addVar(vtype=GRB.INTEGER, lb=0, ub=1, name="x_{}".format(obj))
        )

    objective = 0
    for _criterion in range(n):
        objective += quicksum(
            _vars[i] * utilities[i][_criterion] * _objective_weights[_criterion]
            for i in range(p)
        )
    model.setObjective(objective, GRB.MAXIMIZE)
    model.addConstr(quicksum(_vars[i] * _weights[i] for i in range(p)) <= _b)
    model.setParam("OutputFlag", False)
    model.optimize()
    return [abs(_vars[i].x) for i in range(p)]


if __name__ == "__main__":
    n_objectives = 5
    p_objects = 10

    # Generating random knapsack instance.
    weights, utilities, bound = generate_instance(n=n_objectives, p=p_objects)

    # Create random DM profile.
    weights_dm = create_weights(length=n_objectives, gen_type='ordered')

    # Compute DM best solution
    solution_dm = get_best_solution_dm(weights_dm, weights, utilities, bound)

    additional_constraints = dict()
    niter = 0
    nquestion = 0
    while True:
        niter += 1
        print("{:*^50}".format("Iteration {}".format(niter)))
        m, variables, ipt, npt, ipt_d = get_model(weights, utilities, bound,
                                                  constraints=additional_constraints)

        if ipt == npt:
            model_solution = ipt_d
        else:
            m.optimize()
            model_solution = [abs(variable.x) for variable in variables]

        print("Found solution: {}".format(model_solution))
        print("Solution value: {}".format([sum(model_solution[i] * utilities[i][c]
                                               for i in range(p_objects))
                                           for c in range(n_objectives)]))

        if model_solution == solution_dm:  # DM satisfied.
            print("DM satisfied.")
            break

        nquestion += 1
        print("DM not satisfied.")
        # Find objective with biggest gap.
        worst_gap = 0
        worst_objective = None
        worst_objective_value = None
        for criterion in range(n_objectives):
            solution_dm_value = sum(
                solution_dm[i] * utilities[i][criterion] for i in range(p_objects)
            )
            model_solution_value = sum(
                model_solution[i] * utilities[i][criterion] for i in range(p_objects)
            )
            gap = solution_dm_value - model_solution_value
            gap /= (ipt[criterion] - npt[criterion])
            if gap > worst_gap:
                worst_gap = gap
                worst_objective = criterion
                worst_objective_value = model_solution_value

        m.reset()

        # Add constraint to filter solution space. `worst_objective` must be
        # greater than current value.
        print("Worst objective is {}".format(worst_objective + 1))
        print("Adding constraint z_{} > {}".format(worst_objective + 1,
                                                   worst_objective_value))
        additional_constraints[worst_objective] = worst_objective_value + 1

    print("Found correct solution by asking {} questions.".format(nquestion))
