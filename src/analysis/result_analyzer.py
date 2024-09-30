import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import os
from typing import List, Dict

class ResultAnalyzer:
    """
    Clase para analizar los resultados del backtesting.

    Esta clase proporciona métodos para cargar datos de benchmark, calcular retornos,
    calcular métricas de rendimiento y generar gráficos de los resultados.

    Attributes:
        results (pd.DataFrame): DataFrame con los resultados del backtesting.
        benchmarks (List[str]): Lista de símbolos de benchmark para comparar.
        initial_capital (float): Capital inicial utilizado en el backtesting.
    """

    def __init__(self, results: List[Dict], benchmarks: List[str], initial_capital: float):
        """
        Inicializa el ResultAnalyzer.

        Args:
            results (List[Dict]): Lista de diccionarios con los resultados del backtesting.
            benchmarks (List[str]): Lista de símbolos de benchmark para comparar.
            initial_capital (float): Capital inicial utilizado en el backtesting.
        """
        self.results = pd.DataFrame(results).set_index('Date')
        self.benchmarks = benchmarks
        self.initial_capital = initial_capital
        self.load_benchmark_data()

    def load_benchmark_data(self):
        """
        Carga los datos de los benchmarks especificados y los normaliza al capital inicial.

        Este método descarga los datos históricos de los benchmarks utilizando yfinance
        y los añade al DataFrame de resultados, normalizados al capital inicial.
        """
        for benchmark in self.benchmarks:
            benchmark_data = yf.download(benchmark, start=self.results.index[0], end=self.results.index[-1])['Close']
            benchmark_normalized = benchmark_data / benchmark_data.iloc[0] * self.initial_capital
            self.results[f'{benchmark}_Value'] = benchmark_normalized

    def calculate_returns(self):
        """
        Calcula los retornos de la estrategia y los benchmarks.

        Este método calcula los retornos porcentuales diarios de la estrategia
        y de cada benchmark, y los añade al DataFrame de resultados.
        """
        self.results['Strategy_Returns'] = self.results['Total'].pct_change()
        for benchmark in self.benchmarks:
            self.results[f'{benchmark}_Returns'] = self.results[f'{benchmark}_Value'].ffill().pct_change()

    def calculate_metrics(self):
        """
        Calcula y muestra diversas métricas de rendimiento.

        Este método calcula y muestra las siguientes métricas:
        - Sharpe Ratio
        - Sortino Ratio
        - Maximum Drawdown
        - Total Return
        - Alpha (para cada benchmark)
        - Beta (para cada benchmark)
        - Correlation (para cada benchmark)
        - Information Ratio (para cada benchmark)
        """
        strategy_returns = self.results['Strategy_Returns'].dropna()

        # Sharpe Ratio
        risk_free_rate = 0.02  # Asumimos una tasa libre de riesgo del 2%
        excess_returns = strategy_returns - risk_free_rate / 252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()

        # Sortino Ratio
        downside_returns = excess_returns[excess_returns < 0]
        sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_returns.std()

        # Maximum Drawdown
        cum_returns = (1 + strategy_returns).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # Total Return
        total_return = (self.results['Total'].iloc[-1] / self.results['Total'].iloc[0]) - 1

        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {sortino_ratio:.2f}")
        print(f"Maximum Drawdown: {max_drawdown:.2%}")
        print(f"Total Return: {total_return:.2%}")

        # Métricas para benchmarks
        for benchmark in self.benchmarks:
            benchmark_returns = self.results[f'{benchmark}_Returns'].dropna()
            benchmark_excess_returns = benchmark_returns - risk_free_rate / 252
            
            # Alpha and Beta
            covariance = np.cov(strategy_returns, benchmark_returns)
            beta = covariance[0, 1] / covariance[1, 1]
            alpha = strategy_returns.mean() * 252 - risk_free_rate - beta * (benchmark_returns.mean() * 252 - risk_free_rate)
            
            # Correlation
            correlation = strategy_returns.corr(benchmark_returns)
            
            # Information Ratio
            active_returns = strategy_returns - benchmark_returns
            information_ratio = np.sqrt(252) * active_returns.mean() / active_returns.std()
            
            print(f"\nMétricas para {benchmark}:")
            print(f"Alpha: {alpha:.4f}")
            print(f"Beta: {beta:.2f}")
            print(f"Correlation: {correlation:.2f}")
            print(f"Information Ratio: {information_ratio:.2f}")

    def plot_results(self):
        """
        Genera y guarda dos gráficos de los resultados del backtesting.

        Este método genera dos gráficos:
        1. Rendimiento de la estrategia con señales de compra/venta efectivas.
        2. Rendimiento de la estrategia comparado con los benchmarks.

        Los gráficos se guardan en el directorio 'plots' en la raíz del proyecto.
        """
        # Gráfico 1: Strategy Performance con señales de compra/venta efectivas
        fig1, ax1 = plt.subplots(figsize=(12, 6))

        ax1.plot(self.results.index, self.results['Total'], label='Strategy', linewidth=2)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value')
        ax1.set_title('Strategy Performance with Effective Buy/Sell Signals')

        # Calcular cambios en las tenencias para identificar compras y ventas efectivas
        holdings_change = self.results['Holdings'].diff()

        # Añadir señales de compra y venta efectivas
        buy_signals = self.results[holdings_change > 0].index
        sell_signals = self.results[holdings_change < 0].index

        ax1.scatter(buy_signals, self.results.loc[buy_signals, 'Total'], color='green', marker='^', label='Buy')
        ax1.scatter(sell_signals, self.results.loc[sell_signals, 'Total'], color='red', marker='v', label='Sell')

        ax1.legend(loc='upper left')

        date_formatter = DateFormatter("%Y-%m-%d")
        ax1.xaxis.set_major_formatter(date_formatter)
        fig1.autofmt_xdate()

        plt.tight_layout()

        # Guardar el primer gráfico
        plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        file_path1 = os.path.join(plots_dir, 'strategy_performance.jpg')
        plt.savefig(file_path1, dpi=300, bbox_inches='tight')
        print(f"Gráfico de rendimiento de la estrategia guardado como '{file_path1}'")
        plt.close(fig1)

        # Gráfico 2: Strategy Performance vs Benchmarks
        fig2, ax2 = plt.subplots(figsize=(12, 6))

        ax2.plot(self.results.index, self.results['Total'], label='Strategy', linewidth=2)

        # Plotear benchmarks
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.benchmarks)))
        for benchmark, color in zip(self.benchmarks, colors):
            ax2.plot(self.results.index, self.results[f'{benchmark}_Value'], color=color, label=benchmark, linewidth=1, alpha=0.7)

        ax2.set_xlabel('Date')
        ax2.set_ylabel('Portfolio Value')
        ax2.set_title('Strategy Performance vs Benchmarks')
        ax2.legend(loc='upper left')

        ax2.xaxis.set_major_formatter(date_formatter)
        fig2.autofmt_xdate()

        plt.tight_layout()

        # Guardar el segundo gráfico
        file_path2 = os.path.join(plots_dir, 'strategy_vs_benchmarks.jpg')
        plt.savefig(file_path2, dpi=300, bbox_inches='tight')
        print(f"Gráfico de estrategia vs benchmarks guardado como '{file_path2}'")
        plt.close(fig2)