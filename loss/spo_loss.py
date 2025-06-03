import torch
import numpy as np
from models.portfolio_solver import solve_lp

class SPOLossCalculator:
    def __call__(self, c_hat_tensor, c_true_tensor):
        """
        输入:
            c_hat_tensor: PyTorch Tensor, shape (n,) - 预测的收益向量(即 \hat{c})
            c_true_tensor: PyTorch Tensor, shape (n,) - 真实的收益向量(即 c)
        输出:
            SPO+ loss: PyTorch Tensor
        """
        c_hat_np = c_hat_tensor.detach().cpu().numpy()  # \hat{c}
        c_true_np = c_true_tensor.detach().cpu().numpy()  # c

        # 计算真实收益下的最优解和最优值
        w_star_true = solve_lp(c_true_np)  # w^*(c)
        z_star_true = c_true_np @ w_star_true  # z^*(c) = c^T w^*(c)

        # 构造扰动向量 c - 2 * \hat{c}
        cost_diff = c_true_np - 2 * c_hat_np  # c - 2\hat{c}

        # 计算扰动向量下的最优值 S(c - 2\hat{c})
        LP_term = solve_lp(cost_diff) @ cost_diff  # S(c - 2\hat{c})

        # 计算 2 \hat{c}^T w^*(c)
        second_term = 2 * (c_hat_np @ w_star_true)

        # SPO+ loss
        loss = LP_term + second_term - z_star_true

        return torch.tensor(loss, dtype=torch.float32, requires_grad=True)


