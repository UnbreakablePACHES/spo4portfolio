# ======================================
# Imports & Setup
# ======================================
import torch
import pandas as pd
import numpy as np
import optuna
from torch.optim import Adam
from dateutil.relativedelta import relativedelta

from DataPipeline.DataBuilder import build_dataset
from models.PortfolioModel import PortfolioModelWithFee
from models.LinearInferencer import LinearPredictorTorch
from pyepo.func.surrogate import SPOPlus


# ======================================
# Build Monthly Dataset with Dynamic prev_weight (Oracle path)
# ======================================
def build_monthly_dataset_with_prev(tickers, data_dir, return_df, start_month, num_months, gamma):
    """
    根据训练起始月份和长度，构造月度样本：
        X_month: 每月特征均值，列表长度 = 有效月份数，每个元素 shape [D]
        C_month: 每月真实收益（真 cost，算术收益均值），shape [A]
        prev_list: 每月对应的 prev_weight（上月 oracle 组合），shape [A]

    注意：
    - 这里使用一个单独的 PortfolioModelWithFee 来生成 oracle 路径（prev_weight 序列）
    - 真实收益用每日 log return → arith return 后按月平均
    """
    n_assets = len(tickers)
    optmodel_oracle = PortfolioModelWithFee(n_assets=n_assets, gamma=gamma, budget=1.0)

    X_month = []
    C_month = []
    prev_list = []

    # 初始组合：全 0（全现金）
    prev_w = np.zeros(n_assets)

    for i in range(num_months):
        m_start = start_month + relativedelta(months=i)
        m_end = (m_start + pd.offsets.MonthEnd(0))

        # ===== 特征：当月所有交易日特征的平均 =====
        features_df, _ = build_dataset(
            tickers=tickers,
            data_dir=data_dir,
            start_date=str(m_start.date()),
            end_date=str(m_end.date())
        )
        if features_df.empty:
            # 没有特征数据就跳过这个月
            continue

        features_df.index = pd.to_datetime(features_df.index).normalize()
        x_m = features_df.values.mean(axis=0)  # [D]

        # ===== 真 cost：当月真实算术收益均值 =====
        mask = (return_df.index >= m_start) & (return_df.index <= m_end)
        ret_slice = return_df.loc[mask, tickers]
        if ret_slice.empty:
            continue

        # DailyReturn 文件一般是 log return，这里还原成算术收益
        arith = np.expm1(ret_slice.values)          # [T_days, A]
        c_m = arith.mean(axis=0)                    # [A]

        # 记录样本和 prev_weight
        X_month.append(x_m)
        C_month.append(c_m)
        prev_list.append(prev_w.copy())

        # ===== 用真 cost + 上月仓位求本月 oracle（带手续费） =====
        w_star = optmodel_oracle.optimize(c_m, prev_weight=prev_w)
        prev_w = np.array(w_star)

    return X_month, C_month, prev_list


# ======================================
# Training Function (Monthly, with dynamic prev_weight)
# ======================================
def train_one_epoch_monthly(predictor, X_month, C_month, prev_list,
                            spo_loss_fn, optmodel, optimizer, device):
    predictor.train()
    total_loss = 0.0

    for x_m, c_m, prev_m in zip(X_month, C_month, prev_list):
        # ---- 准备 tensor ----
        x = torch.tensor(x_m, dtype=torch.float32, device=device).unsqueeze(0)         # [1, D]
        c_true = torch.tensor(c_m, dtype=torch.float32, device=device).unsqueeze(0)    # [1, A]

        # ---- 1. 先设定当前样本的 prev_weight ----
        optmodel.set_prev_weight(prev_m)

        # ---- 2. 用「真 cost」算 oracle 解和真实目标值 true_obj ----
        # 注意：optmodel 里用的是真实 c_m（numpy）
        c_true_np = c_m  # 已经是 numpy 数组 [A]
        optmodel.setObj(c_true_np)
        true_sol_np, true_obj_val = optmodel.solve()   # oracle 解 w*(c), obj = c^T w* - gamma*fee

        true_sol = torch.tensor(true_sol_np, dtype=torch.float32, device=device).unsqueeze(0)   # [1, A]
        true_obj = torch.tensor(true_obj_val, dtype=torch.float32, device=device).unsqueeze(0)  # [1]

        # ---- 3. 预测 cost 向量 c_hat ----
        optimizer.zero_grad()
        c_hat = predictor(x)   # [1, A]

        # ---- 4. 调用 SPOPlus：4 个参数版本 ----
        loss = spo_loss_fn(c_hat, c_true, true_sol, true_obj)

        # ---- 5. 反向传播 + 更新 ----
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(X_month)


# ======================================
# Optuna Objective (one rolling training window)
# ======================================
def objective(trial, tickers, return_df, train_start_month, gamma, device):
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    num_epochs = trial.suggest_int("num_epochs", 15, 30)

    num_assets = len(tickers)

    # ===== 构建 12 个月训练集（从 train_start_month 开始） =====
    X_month, C_month, prev_list = build_monthly_dataset_with_prev(
        tickers=tickers,
        data_dir="data/FeatureData",
        return_df=return_df,
        start_month=train_start_month,
        num_months=12,
        gamma=gamma
    )

    if len(X_month) == 0:
        # 防御性：万一数据全空，返回大 loss
        return 1e6

    input_dim = len(X_month[0])

    predictor = LinearPredictorTorch(input_dim, num_assets).to(device)
    optmodel = PortfolioModelWithFee(n_assets=num_assets, gamma=gamma, budget=1.0)
    spo_loss_fn = SPOPlus(optmodel, processes=1, solve_ratio=1.0, reduction="mean")
    optimizer = Adam(predictor.parameters(), lr=lr)

    last_loss = None
    for epoch in range(num_epochs):
        loss = train_one_epoch_monthly(
            predictor, X_month, C_month, prev_list,
            spo_loss_fn, optmodel, optimizer, device
        )
        last_loss = loss

    # 把最优模型参数存起来，后面主循环里用
    trial.set_user_attr("model_state_dict", predictor.state_dict())
    trial.set_user_attr("hyperparams", {"lr": lr, "num_epochs": num_epochs})

    return last_loss


# ======================================
# Main Rolling Loop (Monthly, dynamic prev_weight, with fee)
# ======================================
if __name__ == "__main__":
    tickers = ["EEM", "EFA", "JPXN", "SPY", "XLK", "VTI", "AGG", "DBC"]
    num_assets = len(tickers)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    gamma = 0.003  # 手续费率

    # ===== 加载每日 log return =====
    return_df = pd.read_csv("data/DailyReturn/DailyReturn_8tickers.csv", index_col=0)
    return_df.index = pd.to_datetime(return_df.index).normalize()
    return_df.columns = [col.replace("_return", "") for col in return_df.columns]  # 确保列名和 tickers 对齐

    # ===== 滚动设置 =====
    start_month = pd.to_datetime("2016-01-01")
    n_roll = 108   # 滚动 108 个月

    results = []

    # 回测阶段的“上一月模型组合”（动态 prev_weight）
    prev_month_weight = np.zeros(num_assets)

    for i in range(n_roll):
        infer_start = start_month + relativedelta(months=i)               # 当前要推断的月份（1号）
        train_start = infer_start - relativedelta(years=1)                # 向前推 12 个月作为训练起点
        infer_end = infer_start + pd.offsets.MonthEnd(0)                  # 当月月底

        print(f"\n📅 {infer_start.strftime('%Y-%m')}: "
              f"训练期 {train_start.date()} ~ {(infer_start - pd.Timedelta(days=1)).date()}，"
              f"推断期 {infer_start.date()} ~ {infer_end.date()}")

        # ===== Optuna 超参搜索（针对当前 rolling window） =====
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: objective(
                trial,
                tickers=tickers,
                return_df=return_df,
                train_start_month=train_start,
                gamma=gamma,
                device=device
            ),
            n_trials=8  # 可以调大/调小
        )

        best_trial = study.best_trial
        best_state_dict = best_trial.user_attrs["model_state_dict"]
        best_hparams = best_trial.user_attrs["hyperparams"]
        print(f"  → Best lr={best_hparams['lr']:.2e}, epochs={best_hparams['num_epochs']}")

        # ===== 用最优模型在当月预测 cost，并用带手续费的优化器求组合 =====
        # 构造当月特征（每日），然后按日预测再取平均
        features_df, _ = build_dataset(
            tickers=tickers,
            data_dir="data/FeatureData",
            start_date=str(infer_start.date()),
            end_date=str(infer_end.date())
        )
        features_df.index = pd.to_datetime(features_df.index).normalize()

        if features_df.empty:
            print("⚠️ 当月没有特征数据，跳过。")
            monthly_return = np.nan
            gross_monthly_return = np.nan
            tc = 0.0
            z_star = prev_month_weight.copy()
        else:
            input_dim = features_df.shape[1]

            predictor = LinearPredictorTorch(input_dim, num_assets).to(device)
            predictor.load_state_dict(best_state_dict)
            predictor.eval()

            x_tensor = torch.tensor(features_df.values, dtype=torch.float32, device=device)
            with torch.no_grad():
                c_hat_daily = predictor(x_tensor)           # [T_days, A]
                c_hat = c_hat_daily.mean(dim=0).cpu().numpy()  # [A] 月度 cost 估计

            # ===== 用带手续费优化器求当月组合（动态 prev_weight） =====
            optmodel_infer = PortfolioModelWithFee(n_assets=num_assets, gamma=gamma, budget=1.0)
            optmodel_infer.set_prev_weight(prev_month_weight)
            optmodel_infer.setObj(c_hat)
            z_star, _ = optmodel_infer.solve()
            z_star = np.array(z_star)

            # ===== 计算当月收益（扣手续费） =====
            try:
                # 从日志收益还原算术收益
                arith_return_month = np.expm1(return_df.loc[infer_start:infer_end, tickers].values)
                daily_return = arith_return_month @ z_star
                gross_monthly_return = np.prod(1 + daily_return) - 1

                # 手续费：当月只 rebal 一次
                tc = gamma * np.sum(np.abs(z_star - prev_month_weight))
                monthly_return = gross_monthly_return - tc
            except Exception as e:
                print(f"⚠️ 无法计算 {infer_start.strftime('%Y-%m')} 的组合收益：{e}")
                gross_monthly_return = np.nan
                tc = 0.0
                monthly_return = np.nan

        # 更新 prev_month_weight（模型组合），用于下个月手续费
        prev_month_weight = z_star.copy()

        # 复利累计收益
        prev_cum = 0.0 if i == 0 else results[-1]["CumulativeReturn"]
        cumulative_return = (1 + prev_cum) * (1 + monthly_return) - 1 if not np.isnan(monthly_return) else prev_cum

        results.append({
            "Month": infer_start.strftime("%Y-%m"),
            "PortfolioWeights": list(z_star),
            "GrossMonthlyReturn": gross_monthly_return,
            "TransactionCost": tc,
            "NetMonthlyReturn": monthly_return,
            "CumulativeReturn": cumulative_return
        })

        print(f"组合权重: {np.round(z_star, 3)}，"
              f"毛月收益: {gross_monthly_return:.4f}，"
              f"手续费: {tc:.4f}，"
              f"净月收益: {monthly_return:.4f}，"
              f"累计收益: {cumulative_return:.4f}")

    df_result = pd.DataFrame(results)
    out_path = "result/8_ticker_1ytrain1yinfer/LP+SPO_plus_fee_dynamic_prev.csv"
    df_result.to_csv(out_path, index=False)
    print(f"\n✅ 全部月份处理完成, 结果保存为: {out_path}")
