# portfolio_model.py
import gurobipy as gp
from pyepo.model.grb.grbmodel import optGrbModel  # ← 已确认真实路径
class PortfolioModel(optGrbModel):
    def __init__(self, n_assets, budget=1.0):
        # 初始化 Gurobi 模型
        self._model = gp.Model()
        self.x = self._model.addVars(n_assets, name="x", vtype=gp.GRB.CONTINUOUS)
        self._model.modelSense = gp.GRB.MAXIMIZE
        self._model.addConstr(gp.quicksum(self.x[i] for i in self.x) <= budget)
        super().__init__()
        
    def _getModel(self):
        return self._model, self.x
