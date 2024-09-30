import logging
from typing import List, Dict
import pandas as pd
from ..data.data_loader import DataLoader
from ..strategies.base_strategy import Strategy
from ..portfolio.portfolio import Portfolio

logger = logging.getLogger(__name__)

class Backtester:
    """
    Clase para realizar backtesting de estrategias de trading.

    Esta clase coordina el proceso de backtesting, utilizando un DataLoader para obtener datos históricos,
    una estrategia para generar señales de trading, y un Portfolio para simular las operaciones y el rendimiento.

    Attributes:
        data_loader (DataLoader): Instancia de DataLoader para cargar datos históricos.
        strategy (Strategy): Instancia de la estrategia de trading a probar.
        initial_capital (float): Capital inicial para el backtesting.
        portfolio (Portfolio): Instancia de Portfolio para simular las operaciones.
        results (List[Dict]): Lista para almacenar los resultados del backtesting.
    """

    def __init__(self, symbol: str, start_date: str, end_date: str, strategy: Strategy, initial_capital: float):
        """
        Inicializa el Backtester.

        Args:
            symbol (str): Símbolo del activo a analizar.
            start_date (str): Fecha de inicio del período de backtesting.
            end_date (str): Fecha de fin del período de backtesting.
            strategy (Strategy): Instancia de la estrategia de trading a probar.
            initial_capital (float): Capital inicial para el backtesting.
        """
        required_fields = strategy.required_data()
        self.data_loader = DataLoader(symbol, start_date, end_date, required_fields)
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.portfolio = Portfolio(initial_capital)
        self.results = []

    def run(self):
        """
        Ejecuta el proceso de backtesting.

        Este método carga los datos históricos, aplica la estrategia de trading a cada punto de datos,
        actualiza el portfolio en consecuencia, y registra los resultados.

        Los resultados se almacenan en el atributo 'results' como una lista de diccionarios,
        cada uno conteniendo información sobre el estado del portfolio en cada punto de tiempo.
        """
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