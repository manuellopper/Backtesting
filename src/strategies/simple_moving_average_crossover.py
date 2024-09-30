import numpy as np
import pandas as pd
from typing import Dict, Any, List
from src.strategies.base_strategy import Strategy
import logging

logger = logging.getLogger(__name__)

class SimpleMovingAverageCrossover(Strategy):
    """
    Implementa una estrategia de cruce de medias móviles simples.

    Esta estrategia genera señales de compra cuando la media móvil corta cruza por encima
    de la media móvil larga, y señales de venta cuando la media móvil corta cruza por debajo
    de la media móvil larga.

    Attributes:
        short_window (int): El período para la media móvil corta.
        long_window (int): El período para la media móvil larga.
        short_ma (np.array): Array para almacenar los valores de la media móvil corta.
        long_ma (np.array): Array para almacenar los valores de la media móvil larga.
    """

    def __init__(self, params: Dict[str, Any]):
        """
        Inicializa la estrategia con los parámetros proporcionados.

        Args:
            params (Dict[str, Any]): Un diccionario con los parámetros de la estrategia.
                Debe contener 'short_window' y 'long_window'.
        """
        super().__init__(params)
        self.short_window = self.params['short_window']
        self.long_window = self.params['long_window']
        self.short_ma = np.array([])
        self.long_ma = np.array([])

    @classmethod
    def validate_params(cls, params: Dict[str, Any]):
        """
        Valida los parámetros proporcionados para la estrategia.

        Args:
            params (Dict[str, Any]): Un diccionario con los parámetros a validar.

        Raises:
            ValueError: Si los parámetros no son válidos.
        """
        if 'short_window' not in params or 'long_window' not in params:
            raise ValueError(
                "Los parámetros deben incluir 'short_window' y 'long_window'")
        if params['short_window'] >= params['long_window']:
            raise ValueError("'short_window' debe ser menor que 'long_window'")

    def generate_signal(self, date: pd.Timestamp, data: pd.Series) -> int:
        """
        Genera una señal de trading basada en el cruce de medias móviles.

        Args:
            date (pd.Timestamp): La fecha actual.
            data (pd.Series): Los datos de mercado para la fecha actual.

        Returns:
            int: 1 para una señal de compra, -1 para una señal de venta, 0 para mantener.
        """
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
        """
        Retorna una lista de los campos de datos requeridos por la estrategia.

        Returns:
            List[str]: Una lista con los nombres de los campos requeridos.
        """
        return ['Close']