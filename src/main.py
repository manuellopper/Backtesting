import pandas as pd
import numpy as np
import yfinance as yf
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import logging
import os
import sys 


# Configuración del logging.
def setup_logging(log_to_file=False):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Configurar el logging a la consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Configurar el logging a un archivo si log_to_file es True
    if log_to_file:
        # Obtener el directorio actual
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Subir un nivel y crear la ruta al directorio 'logs'
        logs_dir = os.path.join(os.path.dirname(current_dir), 'logs')

        # Asegurarse de que el directorio 'logs' existe
        os.makedirs(logs_dir, exist_ok=True)

        # Crear la ruta completa del archivo de log
        log_file_path = os.path.join(logs_dir, 'backtest.log')

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Archivo de log creado en: {log_file_path}")

    return logger


# 1. Módulo de Datos (DataLoader)


class DataLoader:
    def __init__(self, symbol: str, start_date: str, end_date: str, required_fields: List[str]):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.required_fields = required_fields
        self.data = None

    def load_data(self):
        logger.info(
            f"Cargando datos para {self.symbol} desde {self.start_date} hasta {self.end_date}")
        all_data = yf.download(
            self.symbol, start=self.start_date, end=self.end_date)

        # Asegurarse de que todos los campos requeridos estén disponibles
        for field in self.required_fields:
            if field not in all_data.columns:
                raise ValueError(
                    f"El campo requerido '{field}' no está disponible en los datos descargados")

        # Seleccionar solo los campos requeridos
        self.data = all_data[self.required_fields]

        if 'Close' in self.required_fields:
            self.data['Returns'] = self.data['Close'].pct_change()

        logger.info(f"Datos cargados: {len(self.data)} filas")

    def get_data_stream(self):
        for index, row in self.data.iterrows():
            yield index, row

# 2. Módulo de Estrategia (Strategy)


class Strategy(ABC):
    def __init__(self, params: Dict[str, Any]):
        self.validate_params(params)
        self.params = params

    @abstractmethod
    def generate_signal(self, date: pd.Timestamp, data: pd.Series) -> int:
        pass

    @classmethod
    def validate_params(cls, params: Dict[str, Any]):
        pass

    @abstractmethod
    def required_data(self) -> List[str]:
        """
        Retorna una lista de los campos de datos requeridos por la estrategia.
        """
        pass

# Estrategia de ejemplo para SPY


class SimpleMovingAverageCrossover(Strategy):
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.short_window = self.params['short_window']
        self.long_window = self.params['long_window']
        self.short_ma = np.array([])
        self.long_ma = np.array([])

    @classmethod
    def validate_params(cls, params: Dict[str, Any]):
        if 'short_window' not in params or 'long_window' not in params:
            raise ValueError(
                "Los parámetros deben incluir 'short_window' y 'long_window'")
        if params['short_window'] >= params['long_window']:
            raise ValueError("'short_window' debe ser menor que 'long_window'")

    def generate_signal(self, date: pd.Timestamp, data: pd.Series) -> int:
        close_price = data['Close']
        self.short_ma = np.append(self.short_ma, close_price)
        self.long_ma = np.append(self.long_ma, close_price)

        if len(self.short_ma) > self.short_window:
            self.short_ma = self.short_ma[-self.short_window:]
        if len(self.long_ma) > self.long_window:
            self.long_ma = self.long_ma[-self.long_window:]

        if len(self.long_ma) < self.long_window:
            return 0

        short_ma_value = np.mean(self.short_ma)
        long_ma_value = np.mean(self.long_ma)

        if short_ma_value > long_ma_value:
            logger.info(f"{date}: Señal de compra generada")
            return 1  # Buy signal
        elif short_ma_value < long_ma_value:
            logger.info(f"{date}: Señal de venta generada")
            return -1  # Sell signal
        else:
            return 0  # Hold

    def required_data(self) -> List[str]:
        return ['Close']

# 3. Módulo de Backtesting (Backtester)


class Backtester:
    def __init__(self, symbol: str, start_date: str, end_date: str, strategy: Strategy, initial_capital: float):
        required_fields = strategy.required_data()
        self.data_loader = DataLoader(
            symbol, start_date, end_date, required_fields)
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.portfolio = Portfolio(initial_capital)
        self.results = []

    def run(self):
        logger.info("Iniciando backtesting")
        self.data_loader.load_data()
        for date, data in self.data_loader.get_data_stream():
            signal = self.strategy.generate_signal(date, data)
            self.portfolio.update(date, data, signal)
            self.results.append({
                'Date': date,
                'Close': data['Close'],
                'Signal': signal,
                'Holdings': self.portfolio.holdings,
                'Cash': self.portfolio.cash,
                'Total': self.portfolio.total_value
            })
        logger.info("Backtesting completado")

# 4. Módulo de Portafolio (Portfolio)


class Portfolio:
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.holdings = 0
        self.total_value = initial_capital

    def update(self, date: pd.Timestamp, data: pd.Series, signal: int):
        price = data['Close']
        if signal == 1 and self.cash >= price:  # Buy
            shares_to_buy = self.cash // price
            self.holdings += shares_to_buy
            self.cash -= shares_to_buy * price
            logger.info(
                f"{date}: Compra de {shares_to_buy} acciones a {price}")
        elif signal == -1 and self.holdings > 0:  # Sell
            self.cash += self.holdings * price
            logger.info(f"{date}: Venta de {self.holdings} acciones a {price}")
            self.holdings = 0

        self.total_value = self.cash + self.holdings * price

# 5. Módulo de Análisis de Resultados (ResultAnalyzer)


class ResultAnalyzer:
    def __init__(self, results: List[Dict], benchmarks: List[str], initial_capital: float):
        self.results = pd.DataFrame(results).set_index('Date')
        self.benchmarks = benchmarks
        self.initial_capital = initial_capital
        self.load_benchmark_data()

    def load_benchmark_data(self):
        for benchmark in self.benchmarks:
            benchmark_data = yf.download(benchmark, start=self.results.index[0], end=self.results.index[-1])['Close']
            # Normalizar el benchmark al capital inicial
            benchmark_normalized = benchmark_data / benchmark_data.iloc[0] * self.initial_capital
            self.results[f'{benchmark}_Value'] = benchmark_normalized

    def calculate_returns(self):
        self.results['Strategy_Returns'] = self.results['Total'].pct_change()
        for benchmark in self.benchmarks:
            self.results[f'{benchmark}_Returns'] = self.results[f'{benchmark}_Value'].pct_change()

    def calculate_metrics(self):
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

        logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        logger.info(f"Sortino Ratio: {sortino_ratio:.2f}")
        logger.info(f"Maximum Drawdown: {max_drawdown:.2%}")
        logger.info(f"Total Return: {total_return:.2%}")

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
            
            logger.info(f"\nMétricas para {benchmark}:")
            logger.info(f"Alpha: {alpha:.4f}")
            logger.info(f"Beta: {beta:.2f}")
            logger.info(f"Correlation: {correlation:.2f}")
            logger.info(f"Information Ratio: {information_ratio:.2f}")

    def plot_results(self):
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
        plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        file_path1 = os.path.join(plots_dir, 'strategy_performance.jpg')
        plt.savefig(file_path1, dpi=300, bbox_inches='tight')
        logger.info(f"Gráfico de rendimiento de la estrategia guardado como '{file_path1}'")
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
        logger.info(f"Gráfico de estrategia vs benchmarks guardado como '{file_path2}'")
        plt.close(fig2)

def run_backtest(symbol: str, start_date: str, end_date: str, strategy: Strategy, initial_capital: float, benchmarks: List[str]):
    logger.info(f"Iniciando backtesting para {symbol} desde {start_date} hasta {end_date}")

    backtester = Backtester(symbol, start_date, end_date, strategy, initial_capital)
    backtester.run()

    analyzer = ResultAnalyzer(backtester.results, benchmarks, initial_capital)
    analyzer.calculate_returns()
    analyzer.calculate_metrics()
    analyzer.plot_results()
    logger.info("Backtesting completado")

def main(log_to_file=False):
    global logger
    logger = setup_logging(log_to_file)
    
    # Configuración del backtest
    symbol = 'SPY'
    start_date = '2010-01-01'
    end_date = '2023-05-31'
    initial_capital = 10000
    benchmarks = ['QQQ', 'IWM']  # Añadimos benchmarks adicionales

    # Creamos una instancia de la estrategia con sus parámetros específicos
    strategy_params = {
        'short_window': 50,
        'long_window': 200
    }
    strategy = SimpleMovingAverageCrossover(strategy_params)

    # Ejecutamos el backtesting
    run_backtest(symbol, start_date, end_date, strategy, initial_capital, benchmarks)

if __name__ == "__main__":
    main(log_to_file=True)