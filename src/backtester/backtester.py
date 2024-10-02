import logging
from typing import List, Dict
import pandas as pd
from ..data.data_loader import DataLoader
from ..strategies.base_strategy import Strategy
from ..portfolio.portfolio import Portfolio

logger = logging.getLogger(__name__)

class Backtester:
    """
    Clase para realizar backtesting de múltiples estrategias de trading simultáneamente.

    Esta clase coordina el proceso de backtesting, utilizando un DataLoader para obtener datos históricos,
    múltiples estrategias para generar señales de trading, y un Portfolio para simular las operaciones y el rendimiento.

    Attributes:
        data_loader (DataLoader): Instancia de DataLoader para cargar datos históricos.
        strategies (List[Strategy]): Lista de instancias de estrategias de trading a probar.
        initial_capital (float): Capital inicial para el backtesting.
        portfolio (Portfolio): Instancia de Portfolio para simular las operaciones.
        results (List[Dict]): Lista para almacenar los resultados del backtesting.
    """

    def __init__(self, symbol: str, start_date: str, end_date: str, strategies: List[Strategy], initial_capital: float):
        """
        Inicializa el Backtester.

        Args:
            symbol (str): Símbolo del activo a analizar.
            start_date (str): Fecha de inicio del período de backtesting.
            end_date (str): Fecha de fin del período de backtesting.
            strategies (List[Strategy]): Lista de instancias de estrategias de trading a probar.
            initial_capital (float): Capital inicial para el backtesting.
        """
        required_fields = set()
        for strategy in strategies:
            required_fields.update(strategy.required_data())
        self.data_loader = DataLoader(symbol, start_date, end_date, list(required_fields))
        self.strategies = strategies
        self.initial_capital = initial_capital
        self.portfolio = Portfolio(initial_capital)
        self.results = []

    def run(self):
        """
        Ejecuta el proceso de backtesting.

        Este método carga los datos históricos, aplica todas las estrategias de trading a cada punto de datos,
        combina las señales, actualiza el portfolio en consecuencia, y registra los resultados.

        Los resultados se almacenan en el atributo 'results' como una lista de diccionarios,
        cada uno conteniendo información sobre el estado del portfolio en cada punto de tiempo.
        """
        logger.info("Iniciando backtesting")
        self.data_loader.load_data()
        for date, data in self.data_loader.get_data_stream():
            signals = [strategy.generate_signal(date, data) for strategy in self.strategies]
            combined_signal = self.combine_signals(signals)
            self.portfolio.update(date, data, combined_signal)
            self.results.append({
                'Date': date,
                'Close': data['Close'],
                'Signal': combined_signal,
                'Holdings': self.portfolio.holdings,
                'Cash': self.portfolio.cash,
                'Total': self.portfolio.total_value
            })
        logger.info("Backtesting completado")

    def combine_signals(self, signals: List[int]) -> int:
        """
        Combina las señales de múltiples estrategias en una sola señal basada en un enfoque democrático.

        Esta función implementa las siguientes reglas:
        1. Si hay alguna señal de compra o venta, se toma la decisión más democrática entre las que indican comprar y las que indican vender.
        2. En caso de empate entre señales de compra y venta, no se hace nada (se devuelve 0).
        3. Si todas las señales son 0 (mantener), se devuelve 0.

        Args:
            signals (List[int]): Lista de señales generadas por las estrategias.
                                Cada señal debe ser -1 (vender), 0 (mantener), o 1 (comprar).

        Returns:
            int: Señal combinada (-1 para vender, 0 para mantener, 1 para comprar).

        Raises:
            ValueError: Si alguna señal en la lista no es -1, 0, o 1.

        Example:
            >>> backtester = Backtester(...)
            >>> backtester.combine_signals([1, -1, 0, 1])
            1
            >>> backtester.combine_signals([1, -1, -1, 1])
            0
            >>> backtester.combine_signals([0, 0, 0, 0])
            0
        """
        # Verificar que todas las señales sean válidas
        if not all(signal in [-1, 0, 1] for signal in signals):
            raise ValueError("Todas las señales deben ser -1, 0, o 1")

        # Contar las señales de compra y venta
        buy_signals = signals.count(1)
        sell_signals = signals.count(-1)

        # Si hay más señales de compra que de venta, comprar
        if buy_signals > sell_signals:
            return 1
        # Si hay más señales de venta que de compra, vender
        elif sell_signals > buy_signals:
            return -1
        # Si hay igual número de señales de compra y venta, o si todas son 0, mantener
        else:
            return 0