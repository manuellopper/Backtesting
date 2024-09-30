from typing import Dict, Any, List
import pandas as pd
from src.strategies.base_strategy import Strategy
import logging

logger = logging.getLogger(__name__)

class BreakoutStrategy(Strategy):
    """
    Implementa una estrategia de breakout para trading.

    Esta estrategia genera señales de compra cuando el precio supera un nivel superior
    y señales de venta cuando cae por debajo de un nivel inferior. Los niveles se calculan
    basándose en los precios históricos y un umbral de breakout.

    Attributes:
        lookback_period (int): Número de períodos para calcular los niveles de breakout.
        breakout_threshold (float): Porcentaje que define el umbral de breakout.
        price_history (list): Lista que almacena los precios históricos recientes.
    """

    def __init__(self, params: Dict[str, Any]):
        """
        Inicializa la estrategia de Breakout.

        Args:
            params (Dict[str, Any]): Diccionario con los parámetros de la estrategia.
                Debe incluir 'lookback_period' y 'breakout_threshold'.
        """
        super().__init__(params)
        self.lookback_period = self.params['lookback_period']
        self.breakout_threshold = self.params['breakout_threshold']
        self.price_history = []

    @classmethod
    def validate_params(cls, params: Dict[str, Any]):
        """
        Valida los parámetros necesarios para la estrategia.

        Args:
            params (Dict[str, Any]): Diccionario con los parámetros a validar.

        Raises:
            ValueError: Si faltan parámetros o si tienen valores inválidos.
        """
        if 'lookback_period' not in params or 'breakout_threshold' not in params:
            raise ValueError("Los parámetros deben incluir 'lookback_period' y 'breakout_threshold'")
        if params['lookback_period'] <= 0 or params['breakout_threshold'] <= 0:
            raise ValueError("'lookback_period' y 'breakout_threshold' deben ser mayores que cero")

    def generate_signal(self, date: pd.Timestamp, data: pd.Series) -> int:
        """
        Genera una señal de trading basada en la estrategia de breakout.

        Args:
            date (pd.Timestamp): Fecha actual.
            data (pd.Series): Datos de mercado para la fecha actual.

        Returns:
            int: 1 para señal de compra, -1 para señal de venta, 0 para mantener.
        """
        close_price = data['Close']
        self.price_history.append(close_price)

        if len(self.price_history) > self.lookback_period:
            self.price_history = self.price_history[-self.lookback_period:]

        if len(self.price_history) < self.lookback_period:
            return 0  # No hay suficientes datos para generar una señal

        recent_high = max(self.price_history[:-1])  # Excluye el precio actual
        recent_low = min(self.price_history[:-1])  # Excluye el precio actual

        # Calcula los niveles de breakout
        upper_breakout = recent_high + (recent_high * self.breakout_threshold)
        lower_breakout = recent_low - (recent_low * self.breakout_threshold)

        if close_price > upper_breakout:
            logger.info(f"{date}: Señal de compra generada (Breakout alcista)")
            return 1  # Señal de compra
        elif close_price < lower_breakout:
            logger.info(f"{date}: Señal de venta generada (Breakout bajista)")
            return -1  # Señal de venta
        else:
            return 0  # Mantener

    def required_data(self) -> List[str]:
        """
        Retorna una lista de los campos de datos requeridos por la estrategia.

        Returns:
            List[str]: Lista de campos requeridos.
        """
        return ['Close']