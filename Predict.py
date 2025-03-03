"""
Predict.py

2025/02/14
股價預測模型用數據處理模組
1. 使用過去5天rolling.mean作為X的生成方式，依照變數特性選擇是否rolling
2. 使用未來一週的報酬作為y
3. 尚未加入行業
4. 只保留每週三的數據進行預測 

"""
import pandas as pd
import numpy as np
import warnings
import importlib
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNetCV, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, log_loss, accuracy_score
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

from sklearn.decomposition import PCA

class MarketNeutralDataProcessor:

    def __init__(self, data, config, industry_dict):
        """
        初始化數據處理器
        :param data: DataFrame, 包含所有日資料
        :param config: dict, 變數名稱的設定，例如：
            {
                'date': 'date',
                'stock_id': 'stock_id',
                'stock_name': 'stock_name',
                'price': 'close',
                'volume': 'volume',
                'no_rolling': ['P/E', 'P/B', 'ROE', 'div_yield'],
                'rolling': ['60vol', '5vol']
            }
        :param industry_dict: dict, 股票名稱對應的行業分類，例如：
            {'台積電': '半導體', '鴻海': '電子', '統一': '食品', ...}
        """
        self.data = data.copy()  # 防止修改原始數據
        self.config = config
        self.industry_dict = industry_dict

    def remove_incomplete_days(self):

        """ 移除當天股票數量小於 50 檔的資料 """
        self.data[self.config['date']] = pd.to_datetime(self.data[self.config['date']])  # 確保日期格式
        daily_stock_count = self.data.groupby(self.config['date'])[self.config['stock_id']].nunique()
        valid_dates = daily_stock_count[daily_stock_count >= 50].index
        self.data = self.data[self.data[self.config['date']].isin(valid_dates)]
        return self.data

    def convert_to_numeric(self):
        """ 確保所有變數為數值型，防止 object 型數據影響機器學習 """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)  # 屏蔽 FutureWarning
            
            for col in self.data.columns:
                if col not in [self.config['date'], self.config['stock_id'], self.config['stock_name']] :  # 排除非數值型變數
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')  # 無法轉換的值轉 NaN
            
            self.data.fillna(0, inplace=True)  # 填補 NaN
            self.data.infer_objects(copy=False)  # 避免 FutureWarning
            self.data = self.data.copy()
        
        return self.data

    def calculate_weekly_return(self):

        """ 計算週報酬 (後5個交易日的報酬) 作為 Y """
        self.data = self.data.sort_values([self.config['stock_id'],
                                           self.config['date']])
        self.data['future_price'] = self.data.groupby(self.config['stock_id'])[self.config['price']].shift(-5)
        self.data.loc[:, 'weekly_return'] = (self.data['future_price'] - self.data[self.config['price']]) / self.data[self.config['price']]
        self.data.loc[:, 'weekly_return_c'] = (self.data['weekly_return'] > 0).astype(int)
        self.data = self.data.copy()

        self.data['prev_price'] = self.data.groupby(self.config['stock_id'])[self.config['price']].shift(5)
        self.data.loc[:, 'prev_weekly_return'] = (self.data[self.config['price']] - self.data['prev_price']) / self.data['prev_price']

        self.data = self.data.copy()

        return self.data
    
    def calculate_weekly_return_wrong(self):

        """ 計算週報酬 (後5個交易日的報酬) 作為 Y """
        self.data = self.data.sort_values([self.config['stock_id'],
                                           self.config['date']])
        self.data['future_price'] = self.data.groupby(self.config['stock_id'])[self.config['price']].shift(5)
        self.data.loc[:, 'weekly_return'] = (self.data[self.config['price']] - self.data['future_price']) / self.data['future_price']
        self.data.loc[:, 'weekly_return_c'] = (self.data['weekly_return'] > 0).astype(int)
        self.data = self.data.copy()

        self.data['prev_price'] = self.data.groupby(self.config['stock_id'])[self.config['price']].shift(5)
        self.data.loc[:, 'prev_weekly_return'] = (self.data[self.config['price']] - self.data['prev_price']) / self.data['prev_price']

        self.data = self.data.copy()

        return self.data
    
    def reduce_column(self):

        """ 整理一些相似 data, 減少變數 """

        self.data = self.data.copy()
        self.data['三大法人買賣價差'] = (self.data['外資賣均價'] - self.data['外資買均價']) + \
                                (self.data['投信賣均價'] - self.data['投信買均價']) + \
                                (self.data['自營商賣均價'] - self.data['自營商買均價'])
        
        self.data['三大法人買賣超金額'] = self.data['外資買賣超金額(千)'] + \
                                        self.data['投信買賣超金額(千)'] + \
                                        self.data['自營商買賣超金額(千)']


    def process_data(self):

        """ 只取最新一期的基本面數據 """
        fundamental_cols = self.config['no_rolling']
        self.data[fundamental_cols] = self.data.groupby(self.config['stock_id'])[fundamental_cols].ffill()
        return self.data

    def process_rolling_data(self):

        """ 取過去5天的波動平均 """
        volatility = self.config['rolling']
        for col in volatility:
            self.data[col] = self.data.groupby(self.config['stock_id'])[col].rolling(window=5, min_periods=1).mean().reset_index(0, drop=True)
        self.data = self.data.copy()
        return self.data

    def add_industry_variable(self):
        """ 加入行業變數 """
        self.data['industry'] = self.data[self.config['stock_name']].map(self.industry_dict)

        # 把行業轉換成 one-hot 編碼
        industry_dummies = pd.get_dummies(self.data['industry'], prefix='industry')
        self.data = pd.concat([self.data, industry_dummies], axis=1)

        # 移除原始的文字型行業變數
        self.data.drop(columns=['industry'], inplace=True)
        self.data = self.data.copy()

        return self.data
    
    def filter_selected_columns(self):
        """ 只保留 `config` 中的欄位 + 行業變數 """
        selected_cols = (
            [self.config['date'], self.config['stock_id'], self.config['stock_name'], 'weekly_return', 'weekly_return_c', 'prev_weekly_return'] +  # 主要標識欄位 & Y
            self.config['no_rolling'] +
            self.config['rolling']
        )
        
        # 加入行業變數（所有 industry_ 開頭的變數）
        industry_cols = [col for col in self.data.columns if col.startswith('industry_')]
        selected_cols += industry_cols

        # 只保留 `selected_cols` 中有的欄位（避免找不到某些欄位）
        self.data = self.data[selected_cols]

        return self.data
    
    def generate_daily_simple_data(self):
        """ 生成簡化版本的 daily data，只包含 `date`, `stock_id`, `stock_name`, `收盤價` """
        simple_data = self.data[[self.config['date'], self.config['stock_id'], 'stock_name', self.config['price']]].copy()
        return simple_data

    def merge_all_data(self):

        """ 最終處理數據，只保留每週五的數據 """
        self.data['day_of_week'] = pd.to_datetime(self.data[self.config['date']]).dt.weekday
        self.data = self.data[self.data['day_of_week'] == 0]  # 只保留每週一
        self.data.drop(columns=['day_of_week'], inplace=True)
        self.data.dropna(inplace=True)  # 確保沒有缺失值
        return self.data

    def process_all(self):

        """ 一鍵處理所有數據 """

        self.remove_incomplete_days()
        self.convert_to_numeric()

        simple_data = self.generate_daily_simple_data()  # 簡化版

        self.reduce_column()
        self.calculate_weekly_return()
        self.process_data()
        self.process_rolling_data()

        self.add_industry_variable()
        self.filter_selected_columns()

        full_data = self.merge_all_data()

        return full_data, simple_data


class RollingWindowModel:
    def __init__(self, full_data, simple_data, model_name, rolling_days=60):
        """
        初始化滾動窗口模型
        :param full_data: 包含完整特徵與標籤的數據
        :param simple_data: 簡化版數據（包含 date, stock_id, stock_name, 收盤價）
        :param model_name: 模型名稱（如 'LinearRegression', 'XGBoost', 'RandomForest', 'Lasso', 'Ridge', 'logistic - L1'）
        :param rolling_days: 滾動窗口週數
        """
        self.full_data = full_data
        self.simple_data = simple_data
        self.model_name = model_name
        self.rolling_days = rolling_days
        self.model = self._initialize_model(model_name)
        self.predictions = None
        self.mse = None

    def _initialize_model(self, model_name):
        """ 根據名稱創建對應的模型 """
        if model_name == "LinearRegression":
            return LinearRegression()
        elif model_name == "Lasso":
            return Lasso(alpha=0.01, max_iter=5000, tol=1e-4)
        elif model_name == "Ridge":
            return Ridge(alpha=1.0)
        elif model_name == "ElasticNet":
            return ElasticNetCV(l1_ratio=np.linspace(0.1, 1, 10), alphas=np.logspace(-4, 2, 100), cv=5)
        elif model_name == "RandomForest":
            return RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        elif model_name == "RandomForestClassifier":
            return RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        elif model_name == "XGBoost":
            return xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1)
        elif model_name == "XGBoostClassifier":
            return xgb.XGBClassifier(objective="binary:logistic", n_estimators=100, learning_rate=0.1)
        elif model_name == 'logistic - L1':
            return LogisticRegression(penalty='l1', solver='liblinear', C=0.1, max_iter=5000)
        else:
            raise ValueError(f"❌ 未知的模型名稱: {model_name}")

    @staticmethod
    def apply_pca(X_train, X_test, n_components=10):
        """
        使用 PCA 降維，手動設置 n_components，確保不會降維過度
        :param X_train: 訓練數據
        :param X_test: 測試數據
        :param n_components: 保留的主成分數量，預設 10
        :return: 降維後的 X_train, X_test, pca_model
        """
        pca = PCA(n_components=n_components)  # 設定降維為 10
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        print(f"🔹 PCA 降維後特徵數: {X_train_pca.shape[1]} (原特徵數: {X_train.shape[1]})")
        
        return X_train_pca, X_test_pca, pca

    
    def model_input(self):

        '''整理模型輸入'''

        self.full_data['date'] = pd.to_datetime(self.full_data['date'])
        self.simple_data['date'] = pd.to_datetime(self.simple_data['date'])

        # 選擇特徵列，排除非數值型變數
        all_feature_cols = [col for col in self.full_data.columns if col not in ['date', 'stock_id', 'stock_name', 'weekly_return', 'weekly_return_c']]
        sorted_data = self.full_data.sort_values(['date', 'stock_id']).reset_index(drop=True)

        unique_dates = sorted_data['date'].unique()
        self.start_date = unique_dates[self.rolling_days]

        return all_feature_cols, sorted_data, unique_dates
    
    def process_predictions(self, predictions, mse_list):
        '''處理預測結果'''
        final_predictions = pd.concat(predictions).reset_index(drop=True)
        avg_mse = np.mean(mse_list)  # 計算平均 MSE 或分類準確率

        # 合併簡化 daily data
        final_predictions = pd.merge(self.simple_data, final_predictions, on=['date', 'stock_id', 'stock_name'], how='outer', suffixes=['', '_1'])

        final_predictions.sort_values(['stock_id', 'date'], inplace=True)
        final_predictions['rank'] = final_predictions.groupby('stock_id')['rank'].bfill(limit=5)
        final_predictions = final_predictions.dropna(subset=['rank'])  # 移除 rank 為 NaN 的行

        # 確保只保留 `start_date` 之後的數據
        final_predictions = final_predictions[final_predictions['date'] >= self.start_date].copy()

        # 儲存預測結果
        self.predictions = final_predictions
        self.mse = avg_mse

        return final_predictions, avg_mse

    def rolling_window_prediction(self):
        """ 使用 60 週滾動窗口訓練模型，前 100 週篩選變數，後續僅保留穩定變數 """

        all_feature_cols, sorted_data, unique_dates = self.model_input()

        predictions = []
        mse_list = []

        # 記錄前 100 週的變數選擇次數
        feature_selection_count = defaultdict(int)

        # 滾動窗口
        for i in range(self.rolling_days, len(unique_dates)):
            train_dates = unique_dates[i - self.rolling_days:i]
            test_date = unique_dates[i]  # 目標預測週

            # 切分訓練集 & 測試集
            train_data = sorted_data[sorted_data['date'].isin(train_dates)]
            test_data = sorted_data[sorted_data['date'] == test_date]

            # 檢查是否為分類模型
            is_classification = self.model_name in ['logistic - L1', 'RandomForestClassifier', 'XGBoostClassifier']

            # 選擇目標變數
            target_col = 'weekly_return_c' if is_classification else 'weekly_return'

            # **變數篩選邏輯**
            if i < 100 + self.rolling_days:
                feature_cols = all_feature_cols  # 前 100 週使用所有變數
            else:
                # 只保留出現超過 50%（至少 50 次）的變數
                feature_cols = [col for col, count in feature_selection_count.items() if count >= 50]
                if not feature_cols:
                    feature_cols = all_feature_cols  # 確保至少有變數可用

            # 分割特徵與標籤
            X_train, y_train = train_data[feature_cols], train_data[target_col]
            X_test, y_test = test_data[feature_cols], test_data[target_col]

            # 訓練模型
            self.model.fit(X_train, y_train)

            # **前 100 週記錄變數選擇情況**
            if i < 100 + self.rolling_days:
                if hasattr(self.model, "coef_"):  # 適用於 Lasso、Ridge、ElasticNet
                    selected_features = [feature_cols[j] for j in range(len(feature_cols)) if abs(self.model.coef_[j]) > 1e-6]
                    selected_features_weights = [self.model.coef_[j] for j in range(len(feature_cols)) if abs(self.model.coef_[j]) > 1e-6]
                    print(selected_features, selected_features_weights)
                elif hasattr(self.model, "feature_importances_"):  # 適用於隨機森林、XGBoost
                    selected_features = [feature_cols[j] for j in range(len(feature_cols)) if self.model.feature_importances_[j] > 0]
                else:
                    selected_features = feature_cols  # 若模型不支持特徵選擇，則使用全部特徵
                
                for feature in selected_features:
                    feature_selection_count[feature] += 1

            self.feature = feature_selection_count

            # 預測並計算評估指標
            if is_classification:
                y_pred_proba = self.model.predict_proba(X_test)[:, 1]  # 取正類的概率
                y_pred_class = self.model.predict(X_test)  # 取預測類別
                accuracy = accuracy_score(y_test, y_pred_class)  # 計算準確率
                mse_list.append(accuracy)  # 記錄分類準確率

                # 儲存預測結果
                test_data = test_data.copy()
                test_data['pred_class'] = y_pred_class
                test_data['pred_proba'] = y_pred_proba
                test_data['rank'] = test_data['pred_proba'].rank(ascending=False, method='first')  # 依照概率排序
            else:
                y_pred = self.model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                mse_list.append(mse)

                # 儲存預測結果
                test_data = test_data.copy()
                test_data['pred_return'] = y_pred
                test_data['rank'] = test_data['pred_return'].rank(ascending=False, method='first')  # 依照回歸值排序

            predictions.append(test_data)

        # 整理最終結果
        final_predictions, avg_mse = self.process_predictions(predictions,
                                                              mse_list)

        # **返回選出的特徵**
        final_selected_features = [col for col, count in feature_selection_count.items() if count >= 50]
        
        return final_predictions, avg_mse, final_selected_features
    
    def rolling_pca_prediction(self):
        """ 使用 PCA 降維 + 簡單線性回歸 進行滾動窗口預測 """
        all_feature_cols, sorted_data, unique_dates = self.model_input()

        predictions = []
        mse_list = []

        # **使用簡單線性回歸**
        base_model = self.model

        # 滾動窗口
        for i in range(self.rolling_days, len(unique_dates)):
            train_dates = unique_dates[i - self.rolling_days:i]
            test_date = unique_dates[i]  # 目標預測週

            # 切分訓練集 & 測試集
            train_data = sorted_data[sorted_data['date'].isin(train_dates)]
            test_data = sorted_data[sorted_data['date'] == test_date]

            # 選擇目標變數
            target_col = 'weekly_return'

            # 分割特徵與標籤
            X_train, y_train = train_data[all_feature_cols], train_data[target_col]
            X_test, y_test = test_data[all_feature_cols], test_data[target_col]

            # **進行 PCA 降維**
            X_train_pca, X_test_pca, pca_model = self.apply_pca(X_train, X_test, n_components=10)

            # 訓練簡單回歸
            base_model.fit(X_train_pca, y_train)

            # **預測並計算 MSE**
            y_pred = base_model.predict(X_test_pca)
            mse = mean_squared_error(y_test, y_pred)
            mse_list.append(mse)

            # 儲存預測結果
            test_data = test_data.copy()
            test_data['pred_return'] = y_pred
            test_data['rank'] = test_data['pred_return'].rank(ascending=False, method='first')

            predictions.append(test_data)

        # 整理最終結果
        final_predictions, avg_mse = self.process_predictions(predictions,
                                                              mse_list)
        # **回傳 PCA 保留的主成分數量**
        n_pca_features = X_train_pca.shape[1]

        return final_predictions, avg_mse, n_pca_features

    def fit_predict(self):

        return self.rolling_window_prediction()

    def run_backtest(self, strategy_module, column="rank", method=1, inventory_df=None):
        """
        執行回測
        :param strategy_module: 回測策略模組（如 MyStrategy）
        :param column: 回測使用的排序列
        :param ton: 是否要計算交易量
        """

        if self.predictions is None:
            print("❌ 請先執行 `fit_predict()` 來獲取預測結果")
            return
        
        # 顯示 MSE
        print(f"模型: {self.model.__class__.__name__}, 平均 MSE: {self.mse:.6f}, 異常週數: {(self.predictions.groupby('date')['rank'].nunique() != 50).sum()}")

        # 重新加載策略模組（防止 Notebook 沒有更新）
        importlib.reload(strategy_module)

        print(f'Running backtest for column: {column} (sell strategy)')
        # 進行回測
        if method == 1:
            strategy = strategy_module.Strategy0050(test_column=column, bos='sell', lookback_period=1)
        elif method == 2:
            strategy = strategy_module.weighted(test_column=column, bos='sell', lookback_period=1)
        elif method == 3:
            strategy = strategy_module.StrategyWithTracking(test_column=column, bos='sell', lookback_period=1)
        elif method == 4:
            strategy = strategy_module.Strategy0050WithIndicators(test_column=column, bos='sell', lookback_period=1)
        
        strategy.backtest(self.predictions)

        # 顯示回測結果
        res = strategy.calculate_performance_metrics()
        print(res)
        strategy.plot_pnl()
        self.lasso_weight = strategy.get_final_day_weights(self.predictions)

        if inventory_df is not None:
            self.trade_count = strategy.generate_trade_dataframe(inventory_df, self.predictions)

    def reg_result(self, check):
        
        check['np1'] = (check['weekly_return'] > 0 )
        check['np2'] = (check['pred_return'] > 0 )
        check['diff'] = (check['np1'] != check['np2'])
        print(f'overall winrate:{1 - check['diff'].mean()}')
        check_plot = check.groupby('date')
        check_plot['diff'].mean().plot()

        loss = []

        for i, j in check_plot:
            diff = j['weekly_return'] - j['pred_return']
            k = (j['diff']).astype(int) * diff.abs()
            loss.append(k)

        loss = np.array(loss)
        print(f'mean loss:{loss.mean()}')


class TuningWindowModel(RollingWindowModel):
    def __init__(self, full_data, simple_data, model_name, rolling_days=60, reuse_model_n_times=1):
        """
        :param reuse_model_n_times: 設定每個模型重複使用的次數，默認為 1
        """
        super().__init__(full_data, simple_data, model_name, rolling_days)
        self.reuse_model_n_times = reuse_model_n_times  # 新增參數
    
    def rolling_window_prediction(self):
        all_feature_cols, sorted_data, unique_dates = self.model_input()
        predictions = []
        mse_list = []
        
        i = self.rolling_days
        while i < len(unique_dates):
            train_dates = unique_dates[i - self.rolling_days:i]
            test_dates = unique_dates[i:i + self.reuse_model_n_times]  # 這次訓練後會預測的日期範圍
            
            train_data = sorted_data[sorted_data['date'].isin(train_dates)]
            test_data = sorted_data[sorted_data['date'].isin(test_dates)]
            
            target_col = 'weekly_return'
            
            X_train, y_train = train_data[all_feature_cols], train_data[target_col]
            X_test, y_test = test_data[all_feature_cols], test_data[target_col]
            
            self.model.fit(X_train, y_train)
            
            if not X_test.empty:
                y_pred = self.model.predict(X_test)
                test_data = test_data.copy()
                test_data['pred_return'] = y_pred
                test_data['rank'] = test_data['pred_return'].rank(ascending=False, method='first')
                predictions.append(test_data)
                
                mse = mean_squared_error(y_test, y_pred) if not y_test.isnull().all() else None
                if mse is not None:
                    mse_list.append(mse)
            
            i += self.reuse_model_n_times  # 每 reuse_model_n_times 週重新訓練一次
        
        return self.process_predictions(predictions, mse_list)
    
    def process_predictions(self, predictions, mse_list):
        '''處理預測結果'''
        final_predictions = pd.concat(predictions).reset_index(drop=True) if predictions else pd.DataFrame()
        avg_mse = np.mean(mse_list) if mse_list else None  # 計算平均 MSE
        
        if not final_predictions.empty:
            final_predictions = pd.merge(self.simple_data, final_predictions, on=['date', 'stock_id', 'stock_name'], how='outer', suffixes=['', '_1'])
            final_predictions.sort_values(['stock_id', 'date'], inplace=True)
            final_predictions['rank'] = final_predictions.groupby('stock_id')['rank'].bfill(limit=5)
            final_predictions = final_predictions.dropna(subset=['rank'])
            final_predictions = final_predictions[final_predictions['date'] >= self.full_data['date'].min()].copy()
        
        return final_predictions, avg_mse
    
    def fit_predict(self):
        '''確保 fit_predict 方法存儲預測結果以供後續使用'''
        self.predictions, self.mse = self.rolling_window_prediction()
        return self.predictions, self.mse

