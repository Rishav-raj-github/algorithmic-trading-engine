"""Momentum Trading Strategy Module

Implements a momentum-based trading strategy using technical indicators
including RSI, MACD, and moving average crossovers.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class MomentumIndicators:
    """Calculate momentum-based technical indicators"""
    
    @staticmethod
    def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index (RSI)
        
        Args:
            prices: Array of closing prices
            period: RSI period (default 14)
            
        Returns:
            RSI values
        """
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = pd.Series(gains).rolling(period).mean()
        avg_loss = pd.Series(losses).rolling(period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.values
    
    @staticmethod
    def calculate_macd(prices: np.ndarray, 
                      fast: int = 12, 
                      slow: int = 26,
                      signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            prices: Array of closing prices
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            Tuple of (MACD, Signal, Histogram)
        """
        prices_series = pd.Series(prices)
        ema_fast = prices_series.ewm(span=fast).mean()
        ema_slow = prices_series.ewm(span=slow).mean()
        
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        
        return macd.values, signal_line.values, histogram.values
    
    @staticmethod
    def calculate_moving_averages(prices: np.ndarray,
                                  short_window: int = 20,
                                  long_window: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate simple moving averages
        
        Args:
            prices: Array of closing prices
            short_window: Short MA period
            long_window: Long MA period
            
        Returns:
            Tuple of (short_MA, long_MA)
        """
        prices_series = pd.Series(prices)
        short_ma = prices_series.rolling(window=short_window).mean().values
        long_ma = prices_series.rolling(window=long_window).mean().values
        return short_ma, long_ma


class MomentumStrategy:
    """Momentum-based trading strategy implementation"""
    
    def __init__(self, 
                 rsi_period: int = 14,
                 rsi_oversold: float = 30,
                 rsi_overbought: float = 70):
        """Initialize momentum strategy
        
        Args:
            rsi_period: RSI calculation period
            rsi_oversold: Oversold threshold
            rsi_overbought: Overbought threshold
        """
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.indicators = MomentumIndicators()
    
    def generate_signals(self, prices: np.ndarray) -> Dict[str, float]:
        """Generate trading signals based on momentum indicators
        
        Args:
            prices: Array of closing prices
            
        Returns:
            Dictionary with signal strengths
        """
        signals = {}
        
        # Calculate RSI
        rsi = self.indicators.calculate_rsi(prices, self.rsi_period)
        if not np.isnan(rsi[-1]):
            signals['rsi'] = rsi[-1]
            signals['rsi_signal'] = self._rsi_signal(rsi[-1])
        
        # Calculate MACD
        macd, signal_line, histogram = self.indicators.calculate_macd(prices)
        if not np.isnan(histogram[-1]):
            signals['macd'] = macd[-1]
            signals['macd_signal'] = 1 if histogram[-1] > 0 else -1
        
        # Calculate Moving Averages
        short_ma, long_ma = self.indicators.calculate_moving_averages(prices)
        if not np.isnan(short_ma[-1]) and not np.isnan(long_ma[-1]):
            signals['ma_crossover'] = 1 if short_ma[-1] > long_ma[-1] else -1
        
        return signals
    
    @staticmethod
    def _rsi_signal(rsi_value: float, 
                   oversold: float = 30,
                   overbought: float = 70) -> int:
        """Convert RSI value to signal
        
        Args:
            rsi_value: Current RSI value
            oversold: Oversold threshold
            overbought: Overbought threshold
            
        Returns:
            1 for buy, -1 for sell, 0 for neutral
        """
        if rsi_value < oversold:
            return 1  # Oversold - buy signal
        elif rsi_value > overbought:
            return -1  # Overbought - sell signal
        return 0  # Neutral
    
    def calculate_entry_exit(self, prices: np.ndarray) -> Tuple[bool, bool]:
        """Calculate entry and exit points
        
        Args:
            prices: Array of closing prices
            
        Returns:
            Tuple of (should_enter, should_exit)
        """
        signals = self.generate_signals(prices)
        
        # Entry: RSI oversold AND MACD bullish
        should_enter = (signals.get('rsi_signal', 0) == 1 and 
                       signals.get('macd_signal', 0) == 1)
        
        # Exit: RSI overbought OR MACD bearish
        should_exit = (signals.get('rsi_signal', 0) == -1 or
                      signals.get('macd_signal', 0) == -1)
        
        return should_enter, should_exit


if __name__ == "__main__":
    # Example usage
    sample_prices = np.random.randn(100).cumsum() + 100
    strategy = MomentumStrategy()
    signals = strategy.generate_signals(sample_prices)
    print(f"Trading Signals: {signals}")
    entry, exit = strategy.calculate_entry_exit(sample_prices)
    print(f"Entry Signal: {entry}, Exit Signal: {exit}")
