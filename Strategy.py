'''
2025/2/12 整理
策略2.0
Strategy.py
1. 上下限設為超參數
2. 差額限制設為超參數
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import seaborn as sns
import matplotlib.dates as mdates


class Strategy0050:

    def __init__(self, initial_capital=0, rebalance_period=5, cost_rate=0.0015,
                 borrow_rate=0.015, tax_fee=0.003, test_column='收盤價',
                 lookback_period=5, bos='buy', num_of_stock=25):
        
        """
        初始化策略參數。

        Args:
            initial_capital (float): 初始資本金額。
            rebalance_period (int): 重新平衡投組的週期（以交易日為單位）。
            cost_rate (float): 交易成本費率。
            borrow_rate (float): 借貸費率（空頭倉位適用）。
            tax_fee (float): 稅率。
            test_column (str): 用於動量計算的欄位（預設為"收盤價"）。
            lookback_period (int): 動量回溯期，決定計算動量時參考的歷史數據範圍。
            bos (str): "buy" 代表做多數值大的，"sell" 代表做空數值大的。
            num_of_stock (int): 多空各持有的股票數量。
        """

        self.initial_capital = initial_capital
        self.rebalance_period = rebalance_period
        self.cost_rate = cost_rate
        self.borrow_rate = borrow_rate
        self.tax_fee = tax_fee
        self.tax_cost = np.nan
        self.test_column = test_column
        self.lookback_period = lookback_period
        self.bos = bos
        self.num_of_stock = num_of_stock
        self.current_capital = initial_capital
        self.weights = {'long': {}, 'short': {}}  # 存储股票权重（按股数）
        self.portfolio = {'long': {}, 'short': {}}  # 存储股票持仓金额
        self.pnl_history = [initial_capital]
        self.long_pnl = [0]
        self.short_pnl = [0]
        self.returns = []
        self.weights_df = pd.DataFrame()

        logging.basicConfig(
            filename="strategy.log",
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
            filemode="w",
            encoding="utf-8"
        )

    def calculate_momentum(self, data):

        """
        計算股票的動量指標，透過回溯期內的價格變化決定強弱排序。

        Args:
            data (pd.DataFrame): 包含股票價格數據的 DataFrame，需包含 "date", "stock_id", "stock_name", 以及測試欄位。

        Returns:
            pd.DataFrame: 根據動量排序的股票清單。
        """
        
        data = data[['date', 'stock_id', 'stock_name', self.test_column]].copy()
        data = data.apply(pd.to_numeric, errors='coerce').fillna(0)
        data['momentum'] = (
            data.groupby('stock_id')[self.test_column]
            .rolling(window=self.lookback_period, min_periods=1)
            .sum().reset_index(level=0, drop=True)
        )
        latest_data = data.groupby('stock_id').tail(1)
        return latest_data.sort_values(by='momentum', ascending=False)

    def calculate_portfolio_weights(self, selected_stocks):
        """
        計算每支股票的持倉權重。

        Args:
            selected_stocks (pd.DataFrame): 已選定的股票清單，需包含 "stock_id" 欄位。

        Returns:
            dict: 股票 ID 與對應的持倉比例。
        """

        weights = {row['stock_id']: 1 / self.num_of_stock for _, row in 
                   selected_stocks.iterrows()}
        logging.info(f"weights=\n{weights}")

        return weights
    
    def ensure_stock_count(self, current_weights, full_rankings, 
                           current_prices):
        """
        確保多空持倉股票數目符合預設值，若不足則補足，若超出則刪減。

        Args:
            current_weights (dict): 當前持倉股票的權重字典。
            full_rankings (pd.DataFrame): 所有可選股票的完整排名表。
            current_prices (dict): 當前市場價格的字典。

        Returns:
            dict: 調整後的持倉權重字典。
        """

        target_count = self.num_of_stock
        # 当前股票数量
        current_count = len(current_weights)

        if current_count < target_count:
            # 补足股票
            missing_count = target_count - current_count
            additional_stocks = full_rankings[~full_rankings['stock_id'].isin(current_weights.keys())].head(missing_count)

            for _, row in additional_stocks.iterrows():
                stock_id = row['stock_id']
                price = current_prices.get(stock_id, 1)
                if price > 0:
                    # 动态计算补足股票的股数（假设剩余金额均匀分配）
                    remaining_value = 9e7 - sum(shares * current_prices.get(stock, 1) for stock, shares in current_weights.items())
                    shares = np.ceil((remaining_value / missing_count) / price / 1000) * 1000
                    current_weights[stock_id] = shares

        elif current_count > target_count:
            # 裁减股票
            excess_count = current_count - target_count
            removable_stocks = sorted(current_weights.keys(), key=lambda x: full_rankings[full_rankings['stock_id'] == x].index[0])

            for stock in removable_stocks[:excess_count]:
                current_weights.pop(stock)

        return current_weights

    def check_positions(self, current_prices, momentum_rankings):
        """
        检查并调整部位金额和股票数量，同时确保多空符合其他条件。
        """
        min_value = 9e7  # 最小金额限制
        max_value = 1e8  # 最大金额限制
        max_difference = 5e6  # 多空差额限制
        max_iterations = 100  # 最大迭代次数
        iteration = 0  # 当前迭代计数

        while iteration < max_iterations:
            iteration += 1

            # 计算当前多头和空头总金额
            long_total_value = sum(
                shares * current_prices.get(stock, 1)
                for stock, shares in self.weights['long'].items()
            )
            short_total_value = sum(
                shares * current_prices.get(stock, 1)
                for stock, shares in self.weights['short'].items()
            )

            # 检查并调整多头金额
            if long_total_value < min_value:
                scaling_factor = min_value / long_total_value if long_total_value > 0 else 1
                for stock in self.weights['long']:
                    self.weights['long'][stock] = np.ceil(self.weights['long'][stock] * scaling_factor / 1000) * 1000

            # 检查并调整空头金额
            if short_total_value < min_value:
                scaling_factor = min_value / short_total_value if short_total_value > 0 else 1
                for stock in self.weights['short']:
                    self.weights['short'][stock] = np.ceil(self.weights['short'][stock] * scaling_factor / 1000) * 1000

            # 确保多空差额不超过限制
            if abs(long_total_value - short_total_value) > max_difference:
                if long_total_value > short_total_value:
                    for stock in sorted(self.weights['long'],
                                        key=self.weights['long'].get,
                                        reverse=True):
                        if self.weights['long'][stock] > 1000:
                            self.weights['long'][stock] -= 1000
                            break
                elif short_total_value > long_total_value:
                    for stock in sorted(self.weights['short'],
                                        key=self.weights['short'].get,
                                        reverse=True):
                        if self.weights['short'][stock] > 1000:
                            self.weights['short'][stock] -= 1000
                            break

            # 确保多空分别有 self.num_of_stock 只股票
            self.weights['long'] = self.ensure_stock_count(
                self.weights['long'], 
                momentum_rankings.head(self.num_of_stock), 
                current_prices
            )
            self.weights['short'] = self.ensure_stock_count(
                self.weights['short'], 
                momentum_rankings.tail(self.num_of_stock).iloc[::-1], 
                current_prices
            )

            # 再次计算多空金额和差额
            long_total_value = sum(
                shares * current_prices.get(stock, 1)
                for stock, shares in self.weights['long'].items()
            )
            short_total_value = sum(
                shares * current_prices.get(stock, 1)
                for stock, shares in self.weights['short'].items()
            )

            # 检查是否满足所有条件
            if (min_value <= long_total_value <= max_value and
                min_value <= short_total_value <= max_value and
                abs(long_total_value - short_total_value) <= max_difference and
                len(self.weights['long']) == self.num_of_stock and
                len(self.weights['short']) == self.num_of_stock):

                break  # 如果所有条件都满足，退出循环

        # 如果达到最大迭代次数，记录日志警告
        if iteration == max_iterations:
            logging.warning(
                "Maximum iterations reached in check_positions. \
                Conditions may not be fully satisfied.")
        
    def _update_weights_df(self, date):
        """
        更新 `self.weights_df`，確保 `stock_id` 為行，`date` 為列，並存儲部位。

        Args:
            date (pd.Timestamp): 當前的調倉日期
        """

        # 將 long / short 權重轉為 DataFrame
        long_df = pd.DataFrame.from_dict(self.portfolio['long'], orient='index', columns=[date])
        short_df = pd.DataFrame.from_dict(self.portfolio['short'], orient='index', columns=[date])

        # **確保所有股票都有記錄，即使當前權重為 0**
        all_stock_ids = set(long_df.index).union(set(short_df.index))

        # 補足缺少的股票，填 0
        long_df = long_df.reindex(all_stock_ids, fill_value=0)
        short_df = short_df.reindex(all_stock_ids, fill_value=0)

        # **計算最終權重（多空差）**
        combined_df = long_df - short_df  # 正數代表多頭，負數代表空頭

        # **更新 `self.weights_df`**
        if self.weights_df.empty:
            self.weights_df = combined_df
        else:
            self.weights_df = self.weights_df.join(combined_df, how='outer')

        logging.info(f"Updated weights_df on {date}:\n{self.weights_df.tail()}")


    def rebalance(self, date, previous_date, data):

        """
        執行投組的再平衡，每週根據動量排序調整多空部位。

        Args:
            date (str or pd.Timestamp): 當前調倉的日期。
            previous_date (str or pd.Timestamp): 上一次調倉的日期。
            data (pd.DataFrame): 包含歷史股價數據的 DataFrame。

        Returns:
            None
        """

        # 保存舊權重
        old_weights = {
            'long': self.weights['long'].copy(),
            'short': self.weights['short'].copy()
        }

        # 计算新的权重
        data["date"] = pd.to_datetime(data["date"])
        previous_date = pd.to_datetime(previous_date)
        lookback_start_date = previous_date - pd.Timedelta(days=self.lookback_period)

        previous_day_data = data[
            (data["date"] <= previous_date) & (data["date"] > lookback_start_date)
        ].copy()
        momentum_rankings = self.calculate_momentum(previous_day_data)
        # 检查动量排名
        logging.info(f"1. Momentum rankings on {date}:\n{momentum_rankings}")

        if self.bos == 'buy':
            top_25 = momentum_rankings.head(self.num_of_stock)
            bottom_25 = momentum_rankings.tail(self.num_of_stock).iloc[::-1]
        elif self.bos == 'sell':
            top_25 = momentum_rankings.tail(self.num_of_stock).iloc[::-1]
            bottom_25 = momentum_rankings.head(self.num_of_stock).iloc[::-1]

        long_weights_ratio = self.calculate_portfolio_weights(top_25)
        short_weights_ratio = self.calculate_portfolio_weights(bottom_25)
        logging.info(f"2. Long weights after rebalance:\n{long_weights_ratio}")
        logging.info(f"2. Short weights after rebalance:\n{short_weights_ratio}")

        # 获取当前价格
        current_day_data = data[data["date"] == date]
        current_day_data.loc[:, '收盤價'] = pd.to_numeric(current_day_data['收盤價'], errors='coerce')
        valid_data = current_day_data[~current_day_data['收盤價'].isna()]
        current_prices = valid_data.set_index("stock_id")["收盤價"].to_dict()
        logging.info(f"3. Current prices on {date}:\n{current_prices}")

        # 计算新的权重（股数）
        long_value = 9e7
        short_value = 9e7

        new_long_weights = {
            stock: np.ceil(long_value * weight / current_prices.get(stock, 1) / 1000) * 1000
            for stock, weight in long_weights_ratio.items()
        }
        new_short_weights = {
            stock: np.ceil(short_value * weight / current_prices.get(stock, 1) / 1000) * 1000
            for stock, weight in short_weights_ratio.items()
        }

        # 更新临时权重
        self.weights['long'] = new_long_weights
        self.weights['short'] = new_short_weights

        logging.info(f"4. Long weights after rebalance:\n{self.weights['long']}")
        logging.info(f"4. Short weights after rebalance:\n{self.weights['short']}")


        # 调用部位检核，确保金额和数量条件
        self.check_positions(current_prices, momentum_rankings)

        # 更新持仓金额
        self.portfolio['long'] = {
            stock: shares * current_prices.get(stock, 1)
            for stock, shares in self.weights['long'].items()
        }
        self.portfolio['short'] = {
            stock: shares * current_prices.get(stock, 1)
            for stock, shares in self.weights['short'].items()
        }
        self._update_weights_df(date)

        # 计算换手率
        turnover = self.calculate_turnover(old_weights, self.weights['long'], self.weights['short'], current_prices)
        self.tax_cost = turnover * (self.tax_fee + self.cost_rate)

        logging.info(f"5. Long value after rebalance:\n{self.portfolio['long']}")
        logging.info(f"5. Short value after rebalance:\n{self.portfolio['short']}")
        logging.info(f"6. {self.test_column} Rebalance on {date}: Turnover: {turnover:.2f}, Tax Cost: {self.tax_cost:.2f}")

    def calculate_turnover(self, old_weights, new_long_weights, new_short_weights, current_prices):
        """
        計算換手率（Turnover），衡量調倉變化的幅度。

        Args:
            old_weights (dict): 調倉前的持倉權重。
            new_long_weights (dict): 新的多頭權重。
            new_short_weights (dict): 新的空頭權重。
            current_prices (dict): 當前市場價格。

        Returns:
            float: 總換手成本（多頭與空頭總計）。
        """
        long_turnover = sum(
            abs((new_long_weights.get(stock, 0) - old_weights['long'].get(stock, 0)) * current_prices.get(stock, 1))
            for stock in set(new_long_weights) | set(old_weights['long'])
        )
        short_turnover = sum(
            abs((new_short_weights.get(stock, 0) - old_weights['short'].get(stock, 0)) * current_prices.get(stock, 1))
            for stock in set(new_short_weights) | set(old_weights['short'])
        )
        # 检查换手率细节
        logging.info(f"7. Old long weights: {old_weights['long']}")
        logging.info(f"7. New long weights: {new_long_weights}")
        logging.info(f"7. Old short weights: {old_weights['short']}")
        logging.info(f"7. New short weights: {new_short_weights}")

        return long_turnover + short_turnover

    def calculate_weekly_pnl(self, data, date, rebalance_date):
        """
        計算單週損益，基於多空倉位變動計算收益。

        Args:
            data (pd.DataFrame): 含股價資訊的 DataFrame。
            date (str or pd.Timestamp): 當前週期的結束日期。
            rebalance_date (str or pd.Timestamp): 上次調倉的日期。

        Returns:
            float: 當週的盈虧金額。
        """

        # 获取当前和上一交易日的价格
        valid_data = data[(data["date"] == date) | (data["date"] == rebalance_date)]
        current_prices = valid_data[valid_data["date"] == date].set_index("stock_id")["收盤價"].to_dict()
        previous_prices = valid_data[valid_data["date"] == rebalance_date].set_index("stock_id")["收盤價"].to_dict()

        # 计算多头收益
        long_pnl = sum(
            self.weights['long'].get(stock, 0) * (current_prices.get(stock, 0) - previous_prices.get(stock, 0))
            for stock in self.weights['long']
        )
        self.long_pnl.append(long_pnl)

        # 计算空头收益
        short_pnl = sum(
            self.weights['short'].get(stock, 0) * (previous_prices.get(stock, 0) - current_prices.get(stock, 0))
            for stock in self.weights['short']
        )
        self.short_pnl.append(short_pnl)
        
        long_total_value = sum(
            self.weights['long'].get(stock, 0) * previous_prices.get(stock, 0)
            for stock in self.weights['long']
        )
        short_total_value = sum(
            self.weights['short'].get(stock, 0) * previous_prices.get(stock, 0)
            for stock in self.weights['short']
        )

        # 避免分母为零
        total_value = long_total_value + short_total_value
        # 计算净收益
        net_pnl = long_pnl + short_pnl - self.tax_cost
        + (short_total_value - long_total_value) * self.borrow_rate

        if total_value == 0:
            net_return = 0
        else:
            net_return = net_pnl / total_value

        # 更新收益记录
        self.returns.append(net_return)

        # 记录日志
        logging.info(f"Long Position Value: {long_total_value:.2f}")
        logging.info(f"Short Position Value: {short_total_value:.2f}")
        logging.info(f"Long PnL: {long_pnl:.2f}, Short PnL: {short_pnl:.2f}, Tax: {self.tax_cost:.2f}")

        return net_pnl

    def calculate_performance_metrics(self):
        """
        計算並回傳績效指標，包括年化報酬率、夏普比率、索提諾比率、勝率、最大回撤、盈虧比等。

        Returns:
            dict: 包含各項績效指標的字典。
                - "Annualized Return": 年化報酬率
                - "Sharpe Ratio": 夏普比率
                - "Sortino Ratio": 索提諾比率
                - "Win Rate": 勝率
                - "Max Drawdown": 最大回撤
                - "Profit/Loss Ratio": 盈虧比
        """
        returns = pd.Series(self.returns)
        # 计算累计收益率和年化收益率
        if returns.empty or len(returns) == 0:
            logging.warning("No returns available for performance metrics.")
            return {
                "Annualized Return": np.nan,
                "Sharpe Ratio": np.nan,
                "Sortino Ratio": np.nan,
                "Win Rate": np.nan,
                "Max Drawdown": np.nan,
                "Profit/Loss Ratio": np.nan
            }

        # **確保 total_days > 0**
        total_days = len(returns) * 5
        if total_days == 0:
            logging.warning("Total trading days is zero. Cannot calculate annualized return.")
            annualized_return = np.nan
        else:
            # **確保 returns.sum() 不會低於 -1**
            total_return = returns.sum()
            if total_return <= -1:
                logging.warning("Total return <= -1, setting annualized return to NaN.")
                annualized_return = np.nan
            else:
                annualized_return = (1 + total_return) ** (252 / total_days) - 1
        self.ar = annualized_return

        # 计算 Sharpe Ratio
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252/7) if returns.std() != 0 else 0

        # 计算 Sortino Ratio
        downside_std = returns[returns < 0].std()
        sortino_ratio = returns.mean() / downside_std * np.sqrt(252/7) if downside_std != 0 else 0

        # 计算胜率
        win_rate = (returns > 0).mean()

        # 计算最大回撤
        max_drawdown = max(
            1 - (self.pnl_history[i] / max(max(self.pnl_history[:i + 1]), 1e-10))
            for i in range(1, len(self.pnl_history))
        )

        # 计算盈亏比
        avg_win = returns[returns > 0].mean()
        avg_loss = -returns[returns < 0].mean()
        profit_loss_ratio = avg_win / avg_loss if avg_loss != 0 else np.inf
        
        logging.info(f"Returns:\n{self.returns}")
        logging.info(f"Annualized Return: {annualized_return:.2f}, Sharpe Ratio: {sharpe_ratio:.2f}")

        return {
            "Annualized Return": annualized_return,
            "Sharpe Ratio": sharpe_ratio,
            "Sortino Ratio": sortino_ratio,
            "Win Rate": win_rate,
            "Max Drawdown": max_drawdown,
            "Profit/Loss Ratio": profit_loss_ratio
        }

    def backtest(self, data):

        """
        執行完整的回測流程，包括：
        1. 計算動量並選股
        2. 執行每週調倉
        3. 計算換手率與交易成本
        4. 記錄每週的 PnL

        Args:
            data (pd.DataFrame): 包含交易數據的 DataFrame。

        Returns:
            None
        """
        data = data.copy()
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date')
        dates = data["date"].unique()

        # 初始化上一次调仓的日期
        last_rebalance_date = None

        for i, date in enumerate(dates):
            if i == 0:
                continue  # 跳过第一天，因为没有前一天数据

            previous_date = dates[i - 1]

            # 每隔 rebalance_period 触发调仓
            if i % self.rebalance_period == 0:
                # 如果已经有一次调仓，计算上一次调仓到当前调仓的 PnL
                if last_rebalance_date is not None:
                    weekly_pnl = self.calculate_weekly_pnl(data, date, last_rebalance_date)
                    self.current_capital += weekly_pnl
                    self.pnl_history.append(self.current_capital)

                # 执行调仓并更新最后一次调仓日期
                self.rebalance(date, previous_date, data)
                last_rebalance_date = date

        # 处理最后一个时间段的 PnL
        if last_rebalance_date is not None and last_rebalance_date != dates[-1]:
            weekly_pnl = self.calculate_weekly_pnl(data, dates[-1], last_rebalance_date)
            self.current_capital += weekly_pnl
            self.pnl_history.append(self.current_capital)

        # 生成 PnL 图表
        #self.plot_pnl()

        # 计算和记录绩效指标
        metrics = self.calculate_performance_metrics()
        logging.info(f"{self.test_column} Performance Metrics:")
        for key, value in metrics.items():
            logging.info(f"{key}: {value:.2f}")

    def plot_pnl(self, save_to_pdf=None):
        """
        繪製 PnL（損益）與累積報酬率圖表，視覺化回測結果。

        Args:
            save_to_pdf (matplotlib.backends.backend_pdf.PdfPages, optional): 若提供則將圖表儲存為 PDF。

        Returns:
            None
        """
        print(self.test_column)
        # 計算累積報酬率
        returns = pd.Series(self.returns)
        cumulative_returns = returns.cumsum()
        # 創建圖形
        fig, axs = plt.subplots(2, 1, figsize=(12, 8))

        # 子圖 1: long short
        l = pd.Series(self.long_pnl).cumsum()
        s = pd.Series(self.short_pnl).cumsum()
        axs[0].plot(l, label='long PnL', color='blue')
        axs[0].plot(s, label='short PnL', color='red')
        axs[0].set_xlabel('Days')
        axs[0].set_ylabel('Portfolio Value')
        axs[0].set_title('PnL Over Time')
        axs[0].legend()
        axs[0].grid()

        # 子圖 2: 累積報酬率
        axs[1].plot(cumulative_returns, label='Cumulative Returns', color='green')
        axs[1].set_xlabel('Rebalance Intervals')
        axs[1].set_ylabel('Cumulative Returns')
        axs[1].set_title('Cumulative Returns Over Time')
        axs[1].legend()
        axs[1].grid()

        # 調整子圖間距
        plt.tight_layout()

        # 保存圖形到 PDF
        if save_to_pdf:
            save_to_pdf.savefig(fig)
            plt.close(fig)
        else:
            plt.show()
        
    # 計算部位
    def get_final_day_weights(self, data):
        """
        取得回測期間最後一天的權重分配，並計算該日的技術指標。

        Args:
            data (pd.DataFrame): 包含股票資訊的 DataFrame。

        Returns:
            tuple: 
                - pd.DataFrame: 最後一天的股票持倉資料。
                - pd.DataFrame: 計算後的技術指標資訊。
        """
         
        # 计算 rolling_value 时不能只选最后一天数据，而是要在完整数据集上计算
        mom = data.copy()
        mom = self.calculate_momentum(mom)
        
        # 获取最后一天的日期
        last_day = data['date'].max()
        
        # 选择最后一天的数据
        new = data[data['date'] == last_day][['date', 'stock_id', 'stock_name', '收盤價']].copy()
        
        # 添加 portfolio 信息
        new['long_position'] = new['stock_id'].map(self.portfolio['long'])
        new['short_position'] = new['stock_id'].map(self.portfolio['short'])

        # 添加 weight 信息
        new['long_weights'] = new['stock_id'].map(self.weights['long'])
        new['short_weights'] = new['stock_id'].map(self.weights['short'])
        
        return new, mom
    
    def generate_trade_dataframe(self, inventory_df, data):
        """
        生成符合下單格式的 DataFrame，固定交易時間為 10:30-12:30。

        Args:
            inventory_df (pd.DataFrame): 當前持倉的 DataFrame，包含：
                - '商品代碼': 股票代號
                - '庫存量張': 持倉張數 (1 張 = 1000 股)
            data (pd.DataFrame): 用於計算目標權重的數據

        Returns:
            pd.DataFrame: 交易單格式的 DataFrame。
        """
        today_date = datetime.today().strftime('%Y/%m/%d')

        trade_start_time = f"{today_date} 10:30"
        trade_end_time = f"{today_date} 12:30"

        # 獲取目標權重
        target_weights, _ = self.get_final_day_weights(data)
        # 確保 long_weights 和 short_weights 不是 NaN
        target_weights['long_weights'] = target_weights['long_weights'].fillna(0)
        target_weights['short_weights'] = target_weights['short_weights'].fillna(0)

        # 轉換持倉資訊
        inventory = dict(zip(inventory_df['商品代碼'], inventory_df['庫存量張']))
        trade_volume = {}

        # 計算應該交易的數量
        for _, row in target_weights.iterrows():
            stock = row['stock_id']
            target_shares = (row['long_weights'] - row['short_weights']) / 1000
            current_shares = inventory.get(stock, 0)
            trade_volume[stock] = target_shares - current_shares  # 負數代表賣出，正數代表買入

        # 生成交易指令 DataFrame
        trade_df = pd.DataFrame(list(trade_volume.items()), columns=['合約月份', '下單總量'])

        # 設定固定值的欄位
        trade_df['開始時間'] = trade_start_time
        trade_df['結束時間'] = trade_end_time
        trade_df['商品型式'] = 'STOCKS'
        trade_df['商品代號'] = 'STOCK'
        trade_df['下單條件'] = 'LIMIT'
        trade_df['價格'] = trade_df['合約月份'].map(lambda x: 'BID_0' if trade_volume[x] < 0 else 'ASK_0')
        trade_df['方向'] = trade_df['合約月份'].map(lambda x: 'SELL' if trade_volume[x] < 0 else 'BUY')
        trade_df['啟動就下'] = 'Y'
        trade_df['每次下單張/口數'] = 2
        trade_df['下單間隔(秒)'] = 10
        trade_df['Portfolio ID'] = 'ENV=SIMU,PID=P2'
        trade_df['零股'] = 'N'

        # 重新排列欄位順序
        trade_df = trade_df[[
            '開始時間', '結束時間', '商品型式', '商品代號', '合約月份', '下單條件', '價格', '方向', '啟動就下',
            '下單總量', '每次下單張/口數', '下單間隔(秒)', 'Portfolio ID', '零股'
        ]]

        # 格式轉換與排序
        trade_df['合約月份'] = trade_df['合約月份'].astype(int)
        trade_df = trade_df.sort_values('下單總量')
        trade_df['下單總量'] = np.abs(trade_df['下單總量'])  # 轉換回張數

        return trade_df

    def plot_heatmap(self, freq="ME"):
        """
        暫存
        生成每個月的平均權重熱力圖
        - X 軸：月份
        - Y 軸：股票 ID
        - 顏色：平均權重（藍色代表多頭，紅色代表空頭）
        """
        weights_df = self.weights_df.T.copy()

        # 按月份計算平均權重
        weights_df.index = pd.to_datetime(weights_df.index, format='%Y-%m-%d')
        monthly_avg_df = weights_df.resample(freq).mean()
        monthly_avg_df.index = pd.to_datetime(monthly_avg_df.index, format='%Y-%m-%d')
        # 繪製熱力圖
        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(monthly_avg_df.T, cmap="bwr", center=0, linewidths=0.5, annot=False)
        plt.xticks(rotation=45)
        plt.title("Monthly Average Position Weight Heatmap")
        plt.xlabel("Month")
        plt.ylabel("Stock ID")
        plt.show()


class weighted(Strategy0050):

    def __init__(self, initial_capital=0, rebalance_period=5, borrow_rate=0.015, tax_fee=0.005,
                 cost_rate=0.0015, test_column="乖離率(250日)",
                 lookback_period=10, bos='buy', num_of_stock=25):
             
        super().__init__(initial_capital=initial_capital,
                         rebalance_period=rebalance_period,
                         borrow_rate=borrow_rate,
                         tax_fee=tax_fee,
                         test_column=test_column,
                         cost_rate=0.0015,
                         lookback_period=lookback_period,
                         bos=bos, num_of_stock=25)
        self.column = test_column

    def calculate_portfolio_weights(self, selected_stocks):
        n_stocks = len(selected_stocks)
        base_weight = 0.7 / (n_stocks - 5) if n_stocks > 5 else 0
        top_weight = 0.3 / min(5, n_stocks)
        weights = {}

        for idx, (_, row) in enumerate(selected_stocks.iterrows()):
            if idx < 5:
                weights[row['stock_id']] = top_weight
            else:
                weights[row['stock_id']] = base_weight

        return weights


class Strategy0050WithIndicators(weighted):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
    
    def apply_technical_filters(self, data):
        # 確保所需欄位存在
        required_columns = ['週ADX(14)', '週+DI(14)', '週-DI(14)', '收盤價']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise KeyError(f"Missing columns in data: {missing_columns}")
    
        
        # 計算技術指標綜合分數 (買入條件 +1, 賣空條件 -1)
        data['tech_score'] = (((data['週ADX(14)']) > 25).astype(int) - ((data['週ADX(14)'])  < 20).astype(int)) * 10 +\
                                (((data['週+DI(14)'] -  data['週-DI(14)'])/3)).astype(float) * 5
        logging.info(f"Applied technical filters:\n{data[['stock_id', 'tech_score']].head()} ")
        return data
    
    def calculate_momentum(self, data):
        data = data[['date', 'stock_id', 'stock_name', self.test_column, 'tech_score']].copy()
        data = data.apply(pd.to_numeric, errors='coerce').fillna(0)
        data['momentum'] = (
            data.groupby('stock_id')[self.test_column]
            .rolling(window=self.lookback_period, min_periods=1).sum().reset_index(level=0, drop=True)
        ) 

        latest_data = data.groupby('stock_id').tail(1)
        return latest_data.sort_values(by='momentum', ascending=False)
    
    def rebalance(self, date, previous_date, data):
        # 先應用技術指標篩選股票
        filtered_data = self.apply_technical_filters(data)
        
        # 進行原本的動量排名
        momentum_rankings = self.calculate_momentum(filtered_data)
        
        logging.info(f"Filtered Momentum rankings on {date}:\n{momentum_rankings}")
        
        # 先確保排名前五和倒數五的股票被選入對應的多空倉位
        top_5 = momentum_rankings.head(5)
        bottom_5 = momentum_rankings.tail(5)
        
        remaining_stocks = momentum_rankings.iloc[5:-5].copy()  # 剩餘的股票
        
        # 根據技術指標門檻設定及動能排名計算綜合分數
        if 'tech_score' not in remaining_stocks.columns:
            raise KeyError("Column 'tech_score' not found in remaining_stocks. Check if apply_technical_filters is applied correctly.")
        remaining_stocks['final_score'] = (
            remaining_stocks['tech_score'] * 2 +  # 增強技術指標影響
            remaining_stocks['momentum'].rank(ascending=True)  # 修正為 ascending=True，讓 momentum 高的股票排名高
        )
        
        # 選出最終的多空倉位股票
        top_25 = pd.concat([top_5, remaining_stocks.nlargest(20, 'final_score')])
        bottom_25 = pd.concat([bottom_5, remaining_stocks.nsmallest(20, 'final_score')])
        
        long_weights_ratio = self.calculate_portfolio_weights(top_25)
        short_weights_ratio = self.calculate_portfolio_weights(bottom_25)
        
        logging.info(f"Long weights after rebalance: {long_weights_ratio}")
        logging.info(f"Short weights after rebalance: {short_weights_ratio}")
        
        # 其他的邏輯與原本的rebalance函數保持一致
        super().rebalance(date, previous_date, data)


class StrategyWithTracking(Strategy0050):
    """
    2025/02/14
    擴展 Strategy0050，增加以下功能：
    1. 紀錄每個 rebalance date 對應的權重變化。
    2. 計算每日 PnL，根據 daily PnL 計算績效並畫圖。
    """

    def __init__(self, *args, **kwargs):
        """
        初始化擴展策略，增加紀錄 rebalance 權重和 daily PnL 的功能。

        Args:
            繼承 Strategy0050 的所有參數。
        """
        super().__init__(*args, **kwargs)

        # 紀錄每次 rebalance 時的權重
        self.rebalance_weights = {}

        # 記錄每日 PnL
        self.daily_pnl = []

    def rebalance(self, date, previous_date, data):
        """
        執行投組的再平衡，並記錄每個 rebalance date 對應的權重。

        Args:
            date (str or pd.Timestamp): 當前調倉的日期。
            previous_date (str or pd.Timestamp): 上一次調倉的日期。
            data (pd.DataFrame): 包含歷史股價數據的 DataFrame。

        Returns:
            None
        """
        # 呼叫原始 rebalance 方法
        super().rebalance(date, previous_date, data)

        # 紀錄當前調倉的權重
        self.rebalance_weights[date] = {
            "long": self.weights['long'].copy(),
            "short": self.weights['short'].copy()
        }
        logging.info(f"Rebalance weights recorded on {date}")

    def calculate_daily_pnl(self, data, date, previous_date, is_rebalance=False):
        """
        計算每日 Long PnL、Short PnL 和 Net PnL，並考慮再平衡日的影響。

        Args:
            data (pd.DataFrame): 包含股票每日收盤價的 DataFrame，需包含 "date" 和 "stock_id"。

        Returns:
            None
        """

        # 取得當前與前一天的價格
        current_prices = data[data["date"] == date].set_index("stock_id")["收盤價"].to_dict()
        previous_prices = data[data["date"] == previous_date].set_index("stock_id")["收盤價"].to_dict()

        # 計算多頭 PnL
        long_pnl = sum(
            self.weights['long'].get(stock, 0) * (current_prices.get(stock, 0) - previous_prices.get(stock, 0))
            for stock in self.weights['long']
        )

        # 計算空頭 PnL
        short_pnl = sum(
            self.weights['short'].get(stock, 0) * (previous_prices.get(stock, 0) - current_prices.get(stock, 0))
            for stock in self.weights['short']
        )

        # 计算多头和空头的总持仓金额
        long_total_value = sum(
            self.weights['long'].get(stock, 0) * previous_prices.get(stock, 0)
            for stock in self.weights['long']
        )
        short_total_value = sum(
            self.weights['short'].get(stock, 0) * previous_prices.get(stock, 0)
            for stock in self.weights['short']
        )

        if is_rebalance:
            net_pnl = long_pnl + short_pnl - self.tax_cost
        else:
            # 記錄每日 PnL（long, short, net）
            net_pnl = long_pnl + short_pnl

        total_value = long_total_value + short_total_value
        if total_value == 0:
            net_return = 0
        else:
            net_return = net_pnl / total_value

        self.returns.append(net_return)
        self.daily_pnl.append({"date": date, "long_pnl": long_pnl, "short_pnl": short_pnl, "net_pnl": net_pnl, 'return': net_return})
        logging.info(f"Daily PnL on {date} - Long: {long_pnl:.2f}, Short: {short_pnl:.2f}, Net: {net_pnl:.2f}, Return: {net_return}")


    def calculate_performance_metrics(self):

        """
        計算並回傳基於 daily PnL 的績效指標。

        Returns:
            dict: 包含各項績效指標的字典。
                - "Annualized Return": 年化報酬率
                - "Sharpe Ratio": 夏普比率
                - "Sortino Ratio": 索提諾比率
                - "Win Rate": 勝率
                - "Max Drawdown": 最大回撤
                - "Profit/Loss Ratio": 盈虧比
        """
        pnl_series = pd.DataFrame(self.daily_pnl)
        pnl_series["cumulative_pnl"] = pnl_series["net_pnl"].cumsum()
        
        returns = pnl_series["return"]

        # 計算年化報酬率

        total_days = len(pnl_series)
        annualized_return = (1 + returns.sum()) ** (252 / total_days) - 1

        # 計算 Sharpe Ratio
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0

        # 計算 Sortino Ratio
        downside_std = returns[returns < 0].std()
        sortino_ratio = returns.mean() / downside_std * np.sqrt(252) if downside_std != 0 else 0

        # 計算勝率
        win_rate = (returns > 0).mean()

        # 計算最大回撤
        max_drawdown = (pnl_series["cumulative_pnl"].cummax() - pnl_series["cumulative_pnl"]).max()

        # 計算盈虧比
        avg_win = returns[returns > 0].mean()
        avg_loss = -returns[returns < 0].mean()
        profit_loss_ratio = avg_win / avg_loss if avg_loss != 0 else np.inf
        
        logging.info(f"Performance metrics based on daily PnL: {annualized_return:.2f}, Sharpe: {sharpe_ratio:.2f}")

        return {
            "Annualized Return": annualized_return,
            "Sharpe Ratio": sharpe_ratio,
            "Sortino Ratio": sortino_ratio,
            "Win Rate": win_rate,
            "Max Drawdown": max_drawdown,
            "Profit/Loss Ratio": profit_loss_ratio
        }

    def plot_pnl(self):
        """
        繪製每日 PnL 和累積 PnL 圖表。

        Returns:
            None
        """

        pnl_series = pd.DataFrame(self.daily_pnl)
        pnl_series["cumulative_pnl"] = pnl_series["net_pnl"].cumsum()

        fig, axs = plt.subplots(2, 1, figsize=(12, 8))

        # 子圖 1: 每日 PnL
        long = pnl_series["long_pnl"].cumsum()
        short = pnl_series["short_pnl"].cumsum()
        axs[0].plot(pnl_series["date"], long, label='long PnL', color='blue')
        axs[0].plot(pnl_series["date"], short, label='short PnL', color='red')
        axs[0].set_xlabel("Date")
        axs[0].set_ylabel("PnL")
        axs[0].set_title("Daily PnL Over Time")
        axs[0].legend()
        axs[0].grid()

        # 子圖 2: 累積 PnL
        axs[1].plot(pnl_series["date"], pnl_series["cumulative_pnl"], label="Cumulative PnL", color="green")
        axs[1].set_xlabel("Date")
        axs[1].set_ylabel("Cumulative PnL")
        axs[1].set_title("Cumulative PnL Over Time")
        axs[1].legend()
        axs[1].grid()

        plt.tight_layout()
        plt.show()

    def backtest(self, data):

        """
        執行完整的回測流程，包括：
        1. 計算動量並選股
        2. 執行每週調倉
        3. 計算換手率與交易成本
        4. 記錄每週的 PnL

        Args:
            data (pd.DataFrame): 包含交易數據的 DataFrame。

        Returns:
            None
        """
        data = data.copy()
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date')
        dates = data["date"].unique()

        # 初始化上一次调仓的日期
        last_rebalance_date = None

        for i, date in enumerate(dates):
            if i == 0:
                previous_date = date
                continue  # 跳过第一天，因为没有前一天数据

            previous_date = dates[i - 1]

            # 每隔 rebalance_period 触发调仓
            if i % self.rebalance_period == 0:
                # 如果已经有一次调仓，计算上一次调仓到当前调仓的 PnL
                if last_rebalance_date is not None:
                    self.calculate_daily_pnl(data, date, previous_date, is_rebalance=True)

                # 执行调仓并更新最后一次调仓日期
                self.rebalance(date, previous_date, data)
                last_rebalance_date = date
            else:
                self.calculate_daily_pnl(data, date, previous_date, is_rebalance=False)


        # 处理最后一个时间段的 PnL
        if last_rebalance_date is not None and last_rebalance_date != dates[-1]:
            self.calculate_daily_pnl(data, dates[-1], dates[-2], is_rebalance=False)


        # 生成 PnL 图表
        #self.plot_pnl()

        # 计算和记录绩效指标
        metrics = self.calculate_performance_metrics()
        logging.info(f"{self.test_column} Performance Metrics:")
        for key, value in metrics.items():
            logging.info(f"{key}: {value:.2f}")


class normalized_weighted(weighted):

    def __init__(self, initial_capital=0, rebalance_period=5,
                 cost_rate=0.0015, borrow_rate=0.015, tax_fee=0.003,
                 test_columns=None, lookback_period=10, bos='buy', num_of_stock=25):
        super().__init__(initial_capital=initial_capital,
                         rebalance_period=rebalance_period,
                         cost_rate=cost_rate,
                         borrow_rate=borrow_rate,
                         tax_fee=tax_fee,
                         test_column=None,
                         lookback_period=lookback_period,
                         bos=bos, num_of_stock=25)
        self.columns = test_columns if test_columns is not None else []

    def normalize_data(self, data):
        # 對指定的多個列進行橫截面標準化，確保數據類型為數值型
        for column in self.columns:
            data[column] = pd.to_numeric(data[column], errors='coerce')
            data[column] = (data[column] - data[column].mean()) / data[column].std()
        return data

    def calculate_standarized_scores(self, data):
        # 確保數據為數值型後計算多列綜合分數
        data[self.columns] = data[self.columns].apply(pd.to_numeric, errors='coerce')
        data['weighted_score'] = data[self.columns].sum(axis=1)
        return data
    
    def calculate_col_momentum(self, data, column):

        """
        計算股票的動量指標，透過回溯期內的價格變化決定強弱排序。

        Args:
            data (pd.DataFrame): 包含股票價格數據的 DataFrame，需包含 "date", "stock_id", "stock_name", 以及測試欄位。

        Returns:
            pd.DataFrame: 根據動量排序的股票清單。
        """
        
        data = data[['date', 'stock_id', 'stock_name', column]].copy()
        data = data.apply(pd.to_numeric, errors='coerce').fillna(0)
        data['momentum'] = (
            data.groupby('stock_id')[column]
            .rolling(window=self.lookback_period, min_periods=1)
            .sum().reset_index(level=0, drop=True)
        )
        latest_data = data.groupby('stock_id').tail(1)
        return latest_data.sort_values(by='stock_id', ascending=False)

    def calculate_momentum(self, data):

        mom = data[['stock_id']].copy()
        mom = mom.sort_values(by='stock_id', ascending=False)

        for column in self.columns:
            # 先計算動量
            x1 = pd.DataFrame(self.calculate_col_momentum(data, column)).copy()
            mom[column] = x1['momentum']

        mom.dropna(axis=0, inplace=True)
        weighted_data = self.calculate_standarized_scores(mom)
        # 根據加權評分進行排序
        ranked_data = weighted_data.sort_values(by='weighted_score', ascending=False)
        return ranked_data


class bas(normalized_weighted):
    
    def __init__(self, initial_capital=0, rebalance_period=5,
                 cost_rate=0.0015, borrow_rate=0.015, tax_fee=0.003,
                 buy_columns=None, sell_columns=None,
                 lookback_period=10, bos='buy', num_of_stock=25):
        super().__init__(initial_capital=initial_capital,
                         rebalance_period=rebalance_period,
                         cost_rate=cost_rate,
                         borrow_rate=borrow_rate,
                         tax_fee=tax_fee,
                         test_columns=None,
                         lookback_period=lookback_period,
                         bos=bos,
                         num_of_stock=25)
        self.bcolumns = buy_columns if buy_columns is not None else {}
        self.scolumns = sell_columns if sell_columns is not None else {}
        self.columns = list(self.bcolumns.keys()) + list(self.scolumns.keys())

    def calculate_standarized_scores(self, data):

        for col in self.columns:
            data[col] = data[col].apply(pd.to_numeric, errors='coerce')

        # 計算買入權重加總
        buy_score = sum(data[col] * weight for col, weight in self.bcolumns.items())

        # 計算賣出權重加總
        sell_score = sum(data[col] * weight for col, weight in self.scolumns.items())

        # 設定買賣規則
        if self.bos == 'buy':
            data['weighted_score'] = buy_score - sell_score
        elif self.bos == 'sell':
            data['weighted_score'] = sell_score - buy_score
        
        return data
    
    def get_final_day_weights(self, data):
        """
        取得回測期間最後一天的權重分配，並計算該日的技術指標。

        Args:
            data (pd.DataFrame): 包含股票資訊的 DataFrame。

        Returns:
            tuple: 
                - pd.DataFrame: 最後一天的股票持倉資料。
                - pd.DataFrame: 計算後的技術指標資訊。
        """
        # 獲取最後一天的日期
        last_day = data['date'].max()

        # 選擇最後一天的數據
        new = data[data['date'] == last_day][['date', 'stock_id', 'stock_name', '收盤價'] + self.columns].copy()

        # 添加 portfolio 信息
        new['long_position'] = new['stock_id'].map(self.portfolio['long']).fillna(0)
        new['short_position'] = new['stock_id'].map(self.portfolio['short']).fillna(0)

        # 添加 weight 信息
        new['long_weights'] = new['stock_id'].map(self.weights['long']).fillna(0)
        new['short_weights'] = new['stock_id'].map(self.weights['short']).fillna(0)

        return new, data

    def generate_trade_dataframe(self, inventory_df):
                
        """
        生成符合下單格式的 DataFrame，固定交易時間為 10:30-12:30。

        Args:
            inventory_df (pd.DataFrame): 當前持倉的 DataFrame，包含：
                - 'stock_id': 股票代號
                - 'quantity': 持倉張數 (1 張 = 1000 股)

        Returns:
            pd.DataFrame: 交易單格式的 DataFrame。
        """
        today_date = datetime.today().strftime('%Y/%m/%d')

        trade_start_time = f"{today_date} 10:30"
        trade_end_time = f"{today_date} 12:30"

        inventory = dict(zip(inventory_df['商品代碼'], inventory_df['庫存量張'] * 1000))
        trade_volume = {}

        # 計算交易量
        for stock, target_shares in {**self.weights['long'], **self.weights['short']}.items():
            current_shares = inventory.get(stock, 0)
            trade_volume[stock] = target_shares - current_shares  # 負數代表賣出，正數代表買入

        # 生成交易指令 DataFrame
        trade_df = pd.DataFrame(list(trade_volume.items()), columns=['合約月份', '下單總量'])
        
        # 設定固定值的欄位
        trade_df['開始時間'] = trade_start_time
        trade_df['結束時間'] = trade_end_time
        trade_df['商品型式'] = 'STOCKS'
        trade_df['商品代號'] = 'STOCK'
        trade_df['下單條件'] = 'LIMIT'
        trade_df['價格'] = trade_df['合約月份'].map(lambda x: 'BID_0' if trade_volume[x] < 0 else 'ASK_0')
        trade_df['方向'] = trade_df['合約月份'].map(lambda x: 'SELL' if trade_volume[x] < 0 else 'BUY')
        trade_df['啟動就下'] = 'Y'
        trade_df['每次下單張/口數'] = 2
        trade_df['下單間隔(秒)'] = 10
        trade_df['Portfolio ID'] = 'ENV=SIMU,PID=P2'
        trade_df['零股'] = 'N'

        # 重新排列欄位順序
        trade_df = trade_df[[
            '開始時間', '結束時間', '商品型式', '商品代號', '合約月份', '下單條件', '價格', '方向', '啟動就下',
            '下單總量', '每次下單張/口數', '下單間隔(秒)', 'Portfolio ID', '零股'
        ]]

        return trade_df
    

class normalized_weighted2(StrategyWithTracking):

    def __init__(self, initial_capital=0, rebalance_period=5,
                 cost_rate=0.0015, borrow_rate=0.015, tax_fee=0.003,
                 test_columns=None, lookback_period=10, bos='buy'):
        super().__init__(initial_capital=initial_capital,
                         rebalance_period=rebalance_period,
                         cost_rate=cost_rate,
                         borrow_rate=borrow_rate,
                         tax_fee=tax_fee,
                         test_column=None,
                         lookback_period=lookback_period,
                         bos=bos,
                         num_of_stock=25)
        self.columns = test_columns if test_columns is not None else []

    def normalize_data(self, data):
        # 對指定的多個列進行橫截面標準化，確保數據類型為數值型
        for column in self.columns:
            data[column] = pd.to_numeric(data[column], errors='coerce')
            data[column] = (data[column] - data[column].mean()) / data[column].std()
        return data

    def calculate_standarized_scores(self, data):
        # 確保數據為數值型後計算多列綜合分數
        data[self.columns] = data[self.columns].apply(pd.to_numeric, errors='coerce')
        data['weighted_score'] = data[self.columns].sum(axis=1)
        return data
    
    def calculate_col_momentum(self, data, column):

        """
        計算股票的動量指標，透過回溯期內的價格變化決定強弱排序。

        Args:
            data (pd.DataFrame): 包含股票價格數據的 DataFrame，需包含 "date", "stock_id", "stock_name", 以及測試欄位。

        Returns:
            pd.DataFrame: 根據動量排序的股票清單。
        """
        
        data = data[['date', 'stock_id', 'stock_name', column]].copy()
        data = data.apply(pd.to_numeric, errors='coerce').fillna(0)
        data['momentum'] = (
            data.groupby('stock_id')[column]
            .rolling(window=self.lookback_period, min_periods=1)
            .sum().reset_index(level=0, drop=True)
        )
        latest_data = data.groupby('stock_id').tail(1)
        return latest_data.sort_values(by='stock_id', ascending=False)

    def calculate_momentum(self, data):

        mom = data[['stock_id']].copy()
        mom = mom.sort_values(by='stock_id', ascending=False)

        for column in self.columns:
            # 先計算動量
            x1 = pd.DataFrame(self.calculate_col_momentum(data, column)).copy()
            mom[column] = x1['momentum']

        mom.dropna(axis=0, inplace=True)
        weighted_data = self.calculate_standarized_scores(mom)
        # 根據加權評分進行排序
        ranked_data = weighted_data.sort_values(by='weighted_score', ascending=False)
        return ranked_data
    

class bas2(normalized_weighted2):
    
    def __init__(self, initial_capital=0, rebalance_period=5,
                 cost_rate=0.0015, borrow_rate=0.015, tax_fee=0.003,
                 buy_columns=None, sell_columns=None,
                 lookback_period=10, bos='buy', num_of_stock=25):
        super().__init__(initial_capital=initial_capital,
                         rebalance_period=rebalance_period,
                         cost_rate=cost_rate,
                         borrow_rate=borrow_rate,
                         tax_fee=tax_fee,
                         test_columns=None,
                         lookback_period=lookback_period,
                         bos=bos,
                         num_of_stock=25)
        self.bcolumns = buy_columns if buy_columns is not None else {}
        self.scolumns = sell_columns if sell_columns is not None else {}
        self.columns = list(self.bcolumns.keys()) + list(self.scolumns.keys())

    def calculate_standarized_scores(self, data):

        for col in self.columns:
            data[col] = data[col].apply(pd.to_numeric, errors='coerce')

        # 計算買入權重加總
        buy_score = sum(data[col] * weight for col, weight in self.bcolumns.items())

        # 計算賣出權重加總
        sell_score = sum(data[col] * weight for col, weight in self.scolumns.items())

        # 設定買賣規則
        if self.bos == 'buy':
            data['weighted_score'] = buy_score - sell_score
        elif self.bos == 'sell':
            data['weighted_score'] = sell_score - buy_score
        
        return data
    
    def get_final_day_weights(self, data):
        """
        取得回測期間最後一天的權重分配，並計算該日的技術指標。

        Args:
            data (pd.DataFrame): 包含股票資訊的 DataFrame。

        Returns:
            tuple: 
                - pd.DataFrame: 最後一天的股票持倉資料。
                - pd.DataFrame: 計算後的技術指標資訊。
        """
        # 獲取最後一天的日期
        last_day = data['date'].max()

        # 選擇最後一天的數據
        new = data[data['date'] == last_day][['date', 'stock_id', 'stock_name', '收盤價'] + self.columns].copy()

        # 添加 portfolio 信息
        new['long_position'] = new['stock_id'].map(self.portfolio['long']).fillna(0)
        new['short_position'] = new['stock_id'].map(self.portfolio['short']).fillna(0)

        # 添加 weight 信息
        new['long_weights'] = new['stock_id'].map(self.weights['long']).fillna(0)
        new['short_weights'] = new['stock_id'].map(self.weights['short']).fillna(0)

        return new, data