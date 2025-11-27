import gurobipy as gp
from pyepo.model.grb.grbmodel import optGrbModel

class PortfolioModel(optGrbModel):
    def __init__(self, n_assets, budget=1.0, ub=None, lb=None):
        self.n_assets = n_assets
        self._model = gp.Model()
        self.x = self._model.addVars(n_assets, name="x", vtype=gp.GRB.CONTINUOUS)
        self._model.modelSense = gp.GRB.MAXIMIZE
        self._model.addConstr(gp.quicksum(self.x[i] for i in self.x) == budget)
        super().__init__()

    def _getModel(self):
        return self._model, self.x

    def setObj(self, cost_vec):
        """设置目标函数:PyEPO 内部会用这个方法"""
        for i, val in enumerate(cost_vec):
            self.x[i].setAttr("obj", val)

    def solve(self):
        """求解优化问题:PyEPO 内部调用时不传参数"""
        self._model.optimize()
        obj_val = self._model.ObjVal
        sol = [self.x[i].X for i in self.x]
        return sol, obj_val
    
    def optimize(self, cost_vec):
        self.setObj(cost_vec)
        sol, _ = self.solve()    # 直接用原本 solve()
        return sol


# ==========================
# 手续费版本（可训练 + 可作为 Oracle）
# 继承 optGrbModel（SPO+ 兼容）
# ==========================
class PortfolioModelWithFee(optGrbModel):
    def __init__(self, n_assets, gamma=0.003, budget=1.0, ub=1.0, lb=0.0):
        self.n_assets = n_assets
        self.gamma = gamma
        self.budget = budget

        # ---- Gurobi model ----
        self._model = gp.Model()
        self._model.setParam("OutputFlag", 0)

        # x: portfolio weights
        self.x = self._model.addVars(
            n_assets, lb=lb, ub=ub, vtype=gp.GRB.CONTINUOUS, name="x"
        )

        # z >= |x - prev|
        self.z = self._model.addVars(
            n_assets, lb=0.0, vtype=gp.GRB.CONTINUOUS, name="z"
        )

        # budget
        self._model.addConstr(
            gp.quicksum(self.x[i] for i in range(n_assets)) == budget
        )

        # initial prev weight = zero portfolio
        self.prev_weight = [0.0] * n_assets

        # store dynamic constraints
        self.z_constrs = []

        super().__init__()

    # PyEPO interface
    def _getModel(self):
        return self._model, self.x

    # update previous period portfolio
    def set_prev_weight(self, prev_weight):
        self.prev_weight = list(prev_weight)

    # SPO+ will call this
    def setObj(self, cost_vec):
        # ============ remove old |x-prev| constraints ============
        for c in self.z_constrs:
            self._model.remove(c)
        self.z_constrs.clear()

        # ============ add new z constraints ============
        for i in range(self.n_assets):
            c1 = self._model.addConstr(self.z[i] >= self.x[i] - self.prev_weight[i])
            c2 = self._model.addConstr(self.z[i] >= -(self.x[i] - self.prev_weight[i]))
            self.z_constrs.extend([c1, c2])

        # ============ objective ============
        ret = gp.quicksum(cost_vec[i] * self.x[i] for i in range(self.n_assets))
        fee = self.gamma * gp.quicksum(self.z[i] for i in range(self.n_assets))
        self._model.setObjective(ret - fee, gp.GRB.MAXIMIZE)

    def solve(self):
        self._model.optimize()
        sol = [self.x[i].X for i in range(self.n_assets)]
        obj = self._model.ObjVal
        return sol, obj

    # oracle usage
    def optimize(self, cost_vec, prev_weight):
        self.set_prev_weight(prev_weight)
        self.setObj(cost_vec)
        self._model.optimize()
        sol = [self.x[i].X for i in range(self.n_assets)]
        return sol



