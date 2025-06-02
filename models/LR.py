# linear_model.py
import pandas as pd
from sklearn.linear_model import LinearRegression

class LinearPredictor:
    def __init__(self, feature_cols=None):
        if feature_cols is None:
            self.feature_cols = ['log_return', 'SMA_10', 'price_bias',
                                 'RSI_14', 'MACD_diff', 'bollinger_width', 'volume_bias']
        else:
            self.feature_cols = feature_cols
        self.model = LinearRegression()

    def load_data(self, csv_path):
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_convert(None)
        self.df = df

    def prepare_target(self, horizon=1):
        df = self.df.copy()
        df['target'] = df['log_return'].shift(-1).rolling(window=horizon).mean()
        df.dropna(subset=self.feature_cols + ['target'], inplace=True)
        self.df = df

    def train(self, start_date, end_date, horizon=1):
        df = self.df
        # 防止用到未来数据（提前 horizon 天）
        end_safe = pd.to_datetime(end_date) - pd.Timedelta(days=horizon)
        train_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_safe)]
        X_train = train_df[self.feature_cols]
        y_train = train_df['target']
        self.model.fit(X_train, y_train)

    def predict(self, start_date, end_date):
        df = self.df
        test_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        X_test = test_df[self.feature_cols]
        y_true = test_df['target'].values
        y_pred = self.model.predict(X_test)
        dates = test_df['Date'].dt.strftime('%Y-%m-%d').tolist()
        return y_pred, y_true, dates
