import logging
import os
import sys
from typing import List

from src.strategies.simple_moving_average_crossover import SimpleMovingAverageCrossover
from src.strategies.breakout_strategy import BreakoutStrategy
from src.backtester.backtester import Backtester
from src.analysis.result_analyzer import ResultAnalyzer
from src.strategies.base_strategy import Strategy

def setup_logging(log_to_file=False):
    """
    Configura el sistema de logging para el framework de backtesting.

    Esta función configura el logging para escribir en la consola y, opcionalmente,
    en un archivo. El nivel de logging se establece en INFO.

    Args:
        log_to_file (bool): Si es True, además de la consola, los logs se escribirán
                            en un archivo en el directorio 'logs'.

    Returns:
        logging.Logger: Un objeto logger configurado.

    Raises:
        OSError: Si hay problemas al crear el directorio de logs o el archivo de log.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_to_file:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.join(os.path.dirname(current_dir), 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        log_file_path = os.path.join(logs_dir, 'backtest.log')

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Archivo de log creado en: {log_file_path}")

    return logger

def run_backtest(symbol: str, start_date: str, end_date: str, strategy: Strategy, initial_capital: float, benchmarks: List[str]):
    """
    Ejecuta un backtesting completo para una estrategia dada.

    Esta función realiza el backtesting de una estrategia de trading para un símbolo
    específico en un período de tiempo determinado. También compara los resultados
    con los benchmarks especificados.

    Args:
        symbol (str): El símbolo del activo a analizar.
        start_date (str): La fecha de inicio del período de backtesting (formato: 'YYYY-MM-DD').
        end_date (str): La fecha de fin del período de backtesting (formato: 'YYYY-MM-DD').
        strategy (Strategy): La estrategia de trading a evaluar.
        initial_capital (float): El capital inicial para el backtesting.
        benchmarks (List[str]): Lista de símbolos de los benchmarks para comparación.

    Returns:
        None

    Raises:
        ValueError: Si los parámetros no son válidos.
    """
    logger.info(f"Iniciando backtesting para {symbol} desde {start_date} hasta {end_date}")

    backtester = Backtester(symbol, start_date, end_date, strategy, initial_capital)
    backtester.run()

    analyzer = ResultAnalyzer(backtester.results, benchmarks, initial_capital)
    analyzer.calculate_returns()
    analyzer.calculate_metrics()
    analyzer.plot_results()
    logger.info("Backtesting completado")

def main(log_to_file=False):
    """
    Función principal que configura y ejecuta el proceso de backtesting con múltiples estrategias.

    Esta función establece los parámetros del backtesting, incluyendo el símbolo,
    las fechas, el capital inicial, los benchmarks y las estrategias a utilizar.
    Luego, ejecuta el backtesting y analiza los resultados.

    Args:
        log_to_file (bool): Si es True, los logs se escribirán también en un archivo.

    Returns:
        None

    Raises:
        Exception: Si ocurre algún error durante la ejecución del backtesting.
    """
    global logger
    logger = setup_logging(log_to_file)
    
    # Configuración del backtest
    symbol = 'SPY'
    start_date = '2015-01-01'
    end_date = '2023-05-31'
    initial_capital = 10000
    benchmarks = ['URTH', 'SPY']

    # Creamos instancias de múltiples estrategias
    sma_strategy = SimpleMovingAverageCrossover({'short_window': 50, 'long_window': 200})
    breakout_strategy = BreakoutStrategy({'lookback_period': 20, 'breakout_threshold': 0.02})

    strategies = [sma_strategy, breakout_strategy]

    # Ejecutamos el backtesting con múltiples estrategias
    backtester = Backtester(symbol, start_date, end_date, strategies, initial_capital)
    backtester.run()

    analyzer = ResultAnalyzer(backtester.results, benchmarks, initial_capital)
    analyzer.calculate_returns()
    analyzer.calculate_metrics()
    analyzer.plot_results()
    logger.info("Backtesting completado")

if __name__ == "__main__":
    main(log_to_file=True)