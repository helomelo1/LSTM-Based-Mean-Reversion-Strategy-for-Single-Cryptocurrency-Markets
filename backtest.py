import numpy as np
import pandas as pd
import vectorbt as vbt
import matplotlib.pyplot as plt


class VectorBTMeanReversion:
    def __init__(self, initial_capital=10000, fees=0.001, slippage=0.0):
        """
        Args:
            initial_capital: Starting capital
            fees: Trading fees (0.001 = 0.1%)
            slippage: Slippage percentage
        """
        self.initial_capital = initial_capital
        self.fees = fees
        self.slippage = slippage
        self.portfolio = None
        self.signals_df = None
    
    def generate_signals(self, df, predictions, z_score_entry=2.0, z_score_exit=0.5, pred_threshold=0.0):
        signals_df = df.copy()
        signals_df['prediction'] = predictions
        
        # Initialize signal columns
        signals_df['long_entry'] = False
        signals_df['short_entry'] = False
        signals_df['long_exit'] = False
        signals_df['short_exit'] = False
        
        # Long entry: Z-score < -threshold AND prediction > 0
        signals_df.loc[
            (signals_df['Z_Score'] < -z_score_entry) & 
            (signals_df['prediction'] > pred_threshold), 
            'long_entry'
        ] = True
        
        # Short entry: Z-score > threshold AND prediction < 0
        signals_df.loc[
            (signals_df['Z_Score'] > z_score_entry) & 
            (signals_df['prediction'] < -pred_threshold), 
            'short_entry'
        ] = True
        
        # Long exit: Z-score crosses *above* the exit threshold
        signals_df['long_exit'] = (signals_df['Z_Score'].shift(1) <= -z_score_exit) & \
                                (signals_df['Z_Score'] > -z_score_exit)

        # Short exit: Z-score crosses *below* the exit threshold
        signals_df['short_exit'] = (signals_df['Z_Score'].shift(1) >= z_score_exit) & \
                                (signals_df['Z_Score'] < z_score_exit)
        
        self.signals_df = signals_df
        
        return signals_df
    
    def run_backtest(self, df, predictions, z_score_entry=2.0, z_score_exit=0.5, pred_threshold=0.0):
        print("\nGenerating trading signals...")
        signals_df = self.generate_signals(df, predictions, z_score_entry, z_score_exit, pred_threshold)
        
        # Create price series
        price = signals_df['Close']
        
        # Generate entries and exits for vectorbt
        # We'll use a simple approach: convert our signals to position changes
        entries = signals_df['long_entry'] | signals_df['short_entry']
        exits = signals_df['long_exit'] | signals_df['short_exit']
        
        # Determine direction: 1 for long, -1 for short
        direction = np.where(signals_df['long_entry'], 1, np.where(signals_df['short_entry'], -1, np.nan))
        direction = pd.Series(direction, index=signals_df.index).fillna(method='ffill').fillna(0).astype(int)
        
        print("Running VectorBT backtest...")
        
        # Create portfolio using VectorBT
        self.portfolio = vbt.Portfolio.from_signals(
            close=price,
            entries=entries,
            exits=exits,
            direction=direction,
            init_cash=self.initial_capital,
            fees=self.fees,
            slippage=self.slippage,
            freq='1h'
        )
        
        print("Backtest completed!")
        
        return self.portfolio
    
    def get_stats(self):
        """Get comprehensive portfolio statistics"""
        if self.portfolio is None:
            raise ValueError("Run backtest first!")
        
        stats = self.portfolio.stats()
        return stats
    
    def print_results(self):
        """Print formatted backtest results"""
        if self.portfolio is None:
            raise ValueError("Run backtest first!")
        
        stats = self.portfolio.stats()
        
        print("\n" + "="*60)
        print("VECTORBT BACKTEST RESULTS")
        print("="*60)
        print(f"Start Value:        ${stats['Start Value']:,.2f}")
        print(f"End Value:          ${stats['End Value']:,.2f}")
        print(f"Total Return:       {stats['Total Return [%]']:.2f}%")
        print(f"Benchmark Return:   {stats.get('Benchmark Return [%]', 0):.2f}%")
        print(f"Max Drawdown:       {stats['Max Drawdown [%]']:.2f}%")
        print(f"Win Rate:           {stats['Win Rate [%]']:.2f}%")
        print(f"Total Trades:       {stats['Total Trades']}")
        print(f"Sharpe Ratio:       {stats.get('Sharpe Ratio', 0):.2f}")
        print(f"Sortino Ratio:      {stats.get('Sortino Ratio', 0):.2f}")
        print(f"Calmar Ratio:       {stats.get('Calmar Ratio', 0):.2f}")
        print(f"Profit Factor:      {stats.get('Profit Factor', 0):.2f}")
        print(f"Expectancy:         ${stats.get('Expectancy', 0):.2f}")
        print("="*60)
    
    def plot_results(self, save_path='vectorbt_backtest_results.html'):
        """
        Plot comprehensive backtest results using VectorBT's built-in plotting.
        
        Args:
            save_path: Path to save HTML plot
        """
        if self.portfolio is None:
            raise ValueError("Run backtest first!")
        
        print(f"\nGenerating interactive plot...")
        
        # Create comprehensive plot
        fig = self.portfolio.plot(subplots=[
            'cum_returns',
            'drawdowns',
            'underwater',
            'trades'
        ])
        
        # Save as HTML (interactive)
        fig.write_html(save_path)
        print(f"Interactive plot saved to '{save_path}'")
        
        # Also create matplotlib version for notebook
        self._plot_matplotlib()
    
    def _plot_matplotlib(self):
        """Create matplotlib plots for Jupyter notebook"""
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        
        # 1. Cumulative returns
        self.portfolio.cum_returns().plot(ax=axes[0], label='Strategy')
        axes[0].set_title('Cumulative Returns')
        axes[0].set_ylabel('Return')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Portfolio value
        self.portfolio.value().plot(ax=axes[1], label='Portfolio Value', color='green')
        axes[1].axhline(y=self.initial_capital, color='red', linestyle='--', alpha=0.5, label='Initial Capital')
        axes[1].set_title('Portfolio Value')
        axes[1].set_ylabel('Value ($)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Drawdown
        self.portfolio.drawdowns().plot(ax=axes[2], color='red', alpha=0.7)
        axes[2].set_title('Drawdown')
        axes[2].set_ylabel('Drawdown')
        axes[2].grid(True, alpha=0.3)
        
        # 4. Underwater plot
        dd = self.portfolio.drawdowns()
        axes[3].fill_between(dd.index, 0, dd.values, color='red', alpha=0.3)
        axes[3].set_title('Underwater Plot (Drawdown from Peak)')
        axes[3].set_ylabel('Drawdown (%)')
        axes[3].set_xlabel('Date')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('vectorbt_backtest_matplotlib.png', dpi=150)
        print("Matplotlib plots saved to 'vectorbt_backtest_matplotlib.png'")
        plt.show()
    
    def get_trades(self):
        """Get detailed trade records"""
        if self.portfolio is None:
            raise ValueError("Run backtest first!")
        
        trades = self.portfolio.trades.records_readable
        return trades
    
    def plot_positions(self, df):
        """Plot price with entry/exit markers"""
        if self.signals_df is None:
            raise ValueError("Generate signals first!")
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Price with signals
        axes[0].plot(df.index, df['Close'], label='ETH Price', alpha=0.7)
        
        # Mark long entries
        long_entries = self.signals_df[self.signals_df['long_entry']]
        axes[0].scatter(long_entries.index, long_entries['Close'], 
                       color='green', marker='^', s=100, label='Long Entry', zorder=5)
        
        # Mark short entries
        short_entries = self.signals_df[self.signals_df['short_entry']]
        axes[0].scatter(short_entries.index, short_entries['Close'], 
                       color='red', marker='v', s=100, label='Short Entry', zorder=5)
        
        axes[0].set_title('Price with Entry Signals')
        axes[0].set_ylabel('Price (USD)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Z-Score with thresholds
        axes[1].plot(df.index, df['Z_Score'], label='Z-Score', color='purple')
        axes[1].axhline(y=2, color='r', linestyle='--', alpha=0.5, label='Entry threshold')
        axes[1].axhline(y=-2, color='r', linestyle='--', alpha=0.5)
        axes[1].axhline(y=0.5, color='g', linestyle='--', alpha=0.5, label='Exit threshold')
        axes[1].axhline(y=-0.5, color='g', linestyle='--', alpha=0.5)
        axes[1].set_title('Z-Score Signals')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Z-Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('vectorbt_signals.png', dpi=150)
        plt.show()
    
    def analyze_returns(self):
        """Analyze return distribution"""
        if self.portfolio is None:
            raise ValueError("Run backtest first!")
        
        returns = self.portfolio.returns()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Return distribution
        axes[0, 0].hist(returns.dropna(), bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Returns')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Return Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Rolling Sharpe
        rolling_sharpe = returns.rolling(window=30*24).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252*24) if x.std() > 0 else 0
        )
        axes[0, 1].plot(rolling_sharpe)
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 1].set_title('Rolling 30-Day Sharpe Ratio')
        axes[0, 1].set_ylabel('Sharpe Ratio')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Cumulative returns
        cum_returns = (1 + returns).cumprod()
        axes[1, 0].plot(cum_returns)
        axes[1, 0].set_title('Cumulative Returns')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Cumulative Return')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Monthly returns heatmap
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns_pivot = monthly_returns.to_frame('returns')
        monthly_returns_pivot['year'] = monthly_returns_pivot.index.year
        monthly_returns_pivot['month'] = monthly_returns_pivot.index.month
        monthly_returns_pivot = monthly_returns_pivot.pivot(index='year', columns='month', values='returns')
        
        import seaborn as sns
        sns.heatmap(monthly_returns_pivot * 100, annot=True, fmt='.2f', cmap='RdYlGn', 
                   center=0, ax=axes[1, 1], cbar_kws={'label': 'Return (%)'})
        axes[1, 1].set_title('Monthly Returns Heatmap (%)')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Year')
        
        plt.tight_layout()
        plt.savefig('vectorbt_returns_analysis.png', dpi=150)
        plt.show()
    
    def optimize_parameters(self, df, predictions, z_score_range, exit_range):
        print("\nRunning parameter optimization...")
        
        results = []
        total_combinations = len(z_score_range) * len(exit_range)
        current = 0
        
        for z_entry in z_score_range:
            for z_exit in exit_range:
                current += 1
                print(f"Testing {current}/{total_combinations}: Z-entry={z_entry}, Z-exit={z_exit}", end='\r')
                
                try:
                    # Generate signals
                    signals_df = self.generate_signals(df, predictions, z_entry, z_exit)
                    price = signals_df['Close']
                    entries = signals_df['long_entry'] | signals_df['short_entry']
                    exits = signals_df['long_exit'] | signals_df['short_exit']
                    direction = np.where(signals_df['long_entry'], 1, 
                                       np.where(signals_df['short_entry'], -1, np.nan))
                    direction = pd.Series(direction, index=signals_df.index).fillna(method='ffill').fillna(0)
                    
                    # Run backtest
                    pf = vbt.Portfolio.from_signals(
                        close=price,
                        entries=entries,
                        exits=exits,
                        direction=direction,
                        init_cash=self.initial_capital,
                        fees=self.fees,
                        freq='1h'
                    )
                    
                    stats = pf.stats()
                    
                    results.append({
                        'z_entry': z_entry,
                        'z_exit': z_exit,
                        'total_return': stats['Total Return [%]'],
                        'sharpe_ratio': stats.get('Sharpe Ratio', 0),
                        'max_drawdown': stats['Max Drawdown [%]'],
                        'total_trades': stats['Total Trades'],
                        'win_rate': stats['Win Rate [%]'],
                        'profit_factor': stats.get('Profit Factor', 0)
                    })
                except Exception as e:
                    continue
        
        print("\nOptimization complete!")
        results_df = pd.DataFrame(results)
        
        # Find best parameters
        best_sharpe = results_df.loc[results_df['sharpe_ratio'].idxmax()]
        best_return = results_df.loc[results_df['total_return'].idxmax()]
        
        print("\nBest parameters by Sharpe Ratio:")
        print(f"  Z-entry: {best_sharpe['z_entry']}, Z-exit: {best_sharpe['z_exit']}")
        print(f"  Sharpe: {best_sharpe['sharpe_ratio']:.2f}, Return: {best_sharpe['total_return']:.2f}%")
        
        print("\nBest parameters by Total Return:")
        print(f"  Z-entry: {best_return['z_entry']}, Z-exit: {best_return['z_exit']}")
        print(f"  Return: {best_return['total_return']:.2f}%, Sharpe: {best_return['sharpe_ratio']:.2f}")
        
        return results_df