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
    def __init__(self, results: List[Dict]):
        self.results = pd.DataFrame(results).set_index('Date')

    def calculate_returns(self):
        self.results['Strategy_Returns'] = self.results['Total'].pct_change()
        self.results['Benchmark_Returns'] = self.results['Close'].pct_change()

    def calculate_metrics(self):
        strategy_returns = self.results['Strategy_Returns'].dropna()
        benchmark_returns = self.results['Benchmark_Returns'].dropna()

        sharpe_ratio = np.sqrt(
            252) * strategy_returns.mean() / strategy_returns.std()
        total_return = (self.results['Total'].iloc[-1] /
                        self.results['Total'].iloc[0]) - 1

        logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        logger.info(f"Total Return: {total_return:.2%}")

    def plot_results(self):

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        ax1.set_ylabel('Portfolio Value', color='tab:blue')
        ax1.plot(self.results.index,
                 self.results['Total'], color='tab:blue', label='Strategy')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2.set_xlabel('Date')
        ax2.set_ylabel('Symbol Price', color='tab:orange')
        ax2.plot(self.results.index,
                 self.results['Close'], color='tab:orange', label='Symbol Price')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        buy_signals = self.results[self.results['Signal'] == 1].index
        sell_signals = self.results[self.results['Signal'] == -1].index

        ax2.scatter(buy_signals, self.results.loc[buy_signals,
                    'Close'], color='green', marker='^', label='Buy Signal')
        ax2.scatter(sell_signals, self.results.loc[sell_signals,
                    'Close'], color='red', marker='v', label='Sell Signal')

        ax1.legend(loc='upper left')
        ax2.legend(loc='upper left')
        plt.title('Backtest Results')

        date_formatter = DateFormatter("%Y-%m-%d")
        ax2.xaxis.set_major_formatter(date_formatter)
        fig.autofmt_xdate()

        plt.tight_layout()

        # Obtener el directorio actual
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Subir un nivel y crear la ruta al directorio 'plots'
        plots_dir = os.path.join(os.path.dirname(current_dir), 'plots')

        # Asegurarse de que el directorio 'plots' existe
        os.makedirs(plots_dir, exist_ok=True)

        # Crear la ruta completa del archivo
        file_path = os.path.join(plots_dir, 'backtest_results.jpg')

        # Guardar el gráfico
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        logger.info(f"Gráfico de resultados guardado como '{file_path}'")
        plt.close()


def run_backtest(symbol: str, start_date: str, end_date: str, strategy: Strategy, initial_capital: float):
    logger.info(
        f"Iniciando backtesting para {symbol} desde {start_date} hasta {end_date}")

    backtester = Backtester(symbol, start_date, end_date,
                            strategy, initial_capital)
    backtester.run()

    analyzer = ResultAnalyzer(backtester.results)
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

    # Creamos una instancia de la estrategia con sus parámetros específicos
    strategy_params = {
        'short_window': 50,
        'long_window': 200
    }
    strategy = SimpleMovingAverageCrossover(strategy_params)

    # Ejecutamos el backtesting
    run_backtest(symbol, start_date, end_date, strategy, initial_capital)


if __name__ == "__main__":
    main(log_to_file=True)
