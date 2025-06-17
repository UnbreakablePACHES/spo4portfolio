import pandas as pd


"""
tickers : Names of assets, list
"""
def build_dataset(tickers, data_dir="data/FeatureData", dropna=True):
    feature_dfs = []
    label_dfs = []

    for ticker in tickers:
        path = f"{data_dir}/{ticker}.csv"
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")

        # 删除多余列
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        df.drop(columns=["Close"], inplace=True, errors="ignore")

        # 使用昨天的 log_return 作为输入特征
        df["log_return_input"] = df["log_return"].shift(1)

        # 构造特征列（包含 log_return_input 和其他技术指标）
        feature_cols = [
            "log_return_input", "SMA_10", "price_bias", "RSI_14",
            "MACD_diff", "bollinger_width", "volume_bias"
        ]
        df_feature = df[feature_cols].copy()
        df_feature.columns = [f"{ticker}_{col}" for col in df_feature.columns]
        feature_dfs.append(df_feature)

        # 构造标签列（当天的 log_return）
        df_label = df[["log_return"]].rename(columns={"log_return": ticker})
        label_dfs.append(df_label)

    # 合并所有 ETF 的特征和标签
    merged_feature = pd.concat(feature_dfs, axis=1, join="inner")
    merged_label = pd.concat(label_dfs, axis=1, join="inner")

    if dropna:
        merged_feature.dropna(inplace=True)
        merged_label = merged_label.loc[merged_feature.index]

    return merged_feature, merged_label

