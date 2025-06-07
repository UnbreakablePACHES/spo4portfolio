import gurobipy as gp
from pyepo.model.grb.grbmodel import optGrbModel

class PortfolioModel(optGrbModel):
    def __init__(self, n_assets, budget=1.0, ub=None, lb=None):
        self.n_assets = n_assets
        self._model = gp.Model()
        self.x = self._model.addVars(n_assets, name="x", vtype=gp.GRB.CONTINUOUS)
        self._model.modelSense = gp.GRB.MAXIMIZE
        self._model.addConstr(gp.quicksum(self.x[i] for i in self.x) <= budget)
        super().__init__()

    def _getModel(self):
        return self._model, self.x

    def setObj(self, cost_vec):
        """设置目标函数：PyEPO 内部会用这个方法"""
        for i, val in enumerate(cost_vec):
            self.x[i].setAttr("obj", val)

    def solve(self):
        """求解优化问题：PyEPO 内部调用时不传参数"""
        self._model.optimize()
        obj_val = self._model.ObjVal
        sol = [self.x[i].X for i in self.x]
        return sol, obj_val



