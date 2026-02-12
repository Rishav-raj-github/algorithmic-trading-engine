"""Backtesting Framework Module

Comprehensive backtesting framework for testing trading strategies
against historical market data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from a backtest run"""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    consecutive_losses: int
    max_consecutive_losses: int


class BacktestEngine:
    """Main backtesting engine"""
    
    def __init__(self, initial_capital: float = 100000,
                 transaction_cost: float = 0.001,
                 slippage: float = 0.0005):
        """Initialize backtest engine
        
        Args:
            initial_capital: Starting capital
            transaction_cost: Cost per transaction (0.1%)
            slippage: Price slippage (0.05%)
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = [initial_capital]
        self.returns: List[float] = []
        
    def backtest(self, prices: pd.Series, 
                 signals: pd.Series,
                 position_size: float = 1.0) -> BacktestResult:
        """Run backtest on price data with trading signals
        
        Args:
            prices: Series of closing prices
            signals: Series of trading signals (1 for buy, -1 for sell, 0 for hold)
            position_size: Fraction of capital to risk per trade
            
        Returns:
            BacktestResult with performance metrics
        """
        capital = self.initial_capital
        position = 0
        entry_price = 0
        
        for i in range(1, len(prices)):
            signal = signals.iloc[i] if i < len(signals) else 0
            price = prices.iloc[i]
            
            # Entry signal
            if signal == 1 and position == 0:
                entry_price = price * (1 + self.slippage)
                position = (capital * position_size) / entry_price
                cost = position * entry_price * (1 + self.transaction_cost)
                capital -= cost
                self.trades.append({
                    'entry_date': prices.index[i],
                    'entry_price': entry_price,
                    'type': 'BUY'
                })
            
            # Exit signal
            elif signal == -1 and position > 0:
                exit_price = price * (1 - self.slippage)
                proceeds = position * exit_price * (1 - self.transaction_cost)
                capital += proceeds
                self.trades[-1]['exit_date'] = prices.index[i]
                self.trades[-1]['exit_price'] = exit_price
                self.trades[-1]['profit'] = proceeds - (position * entry_price)
                position = 0
            
            # Update equity curve
            if position > 0:
                equity = capital + (position * price)
            else:
                equity = capital
            self.equity_curve.append(equity)
            self.returns.append((equity - self.equity_curve[-2]) / self.equity_curve[-2])
        
        return self._calculate_metrics(capital)
    
    def _calculate_metrics(self, final_capital: float) -> BacktestResult:
        """Calculate performance metrics"""
        returns = np.array(self.returns)
        
        total_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        cumulative_returns = np.cumprod(1 + returns) - 1
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / (1 + running_max)
        max_dd = np.min(drawdown) if len(drawdown) > 0 else 0
        
        profits = [t.get('profit', 0) for t in self.trades if 'profit' in t]
        winning = [p for p in profits if p > 0]
        losing = [p for p in profits if p < 0]
        
        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=len(winning) / len(profits) if profits else 0,
            profit_factor=sum(winning) / abs(sum(losing)) if losing else 0,
            total_trades=len(self.trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            avg_win=np.mean(winning) if winning else 0,
            avg_loss=np.mean(losing) if losing else 0,
            consecutive_losses=0,
            max_consecutive_losses=0
        )


if __name__ == "__main__":
    # Example backtest
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    prices = pd.Series(np.random.randn(252).cumsum() + 100, index=dates)
    signals = pd.Series([0] * 252, index=dates)
    
    engine = BacktestEngine(initial_capital=100000)
    result = engine.backtest(prices, signals)
    print(f"Backtest Result: {result}")
