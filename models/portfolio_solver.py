import cvxpy as cp
import numpy as np

def solve_lp(c_vec):
    """
    输入：
        c_vec: 1D numpy array，预测或真实的收益向量
    返回：
        w_opt: 最优解 w (numpy array), 满足：
               max c^T w, s.t. sum(w) = 1, w >= 0
    """
    n = len(c_vec)
    w = cp.Variable(n)
    objective = cp.Maximize(c_vec @ w)
    constraints = [cp.sum(w) == 1, w >= 0]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return w.value
