import pandas as pd
import numpy as np
from typing import Dict, Any, List
from .base_strategy import Strategy

class SmaRsiBollingerStrategy(Strategy):
    """
    Estrategia de trading basada en el cruce de Medias Móviles, RSI y Bandas de Bollinger.
    """

    def __init__(self, params: Dict[str, Any]):
        """
        Inicializa la estrategia con los parámetros proporcionados.

        Args:
            params (Dict[str, Any]): Diccionario con los parámetros de la estrategia.
        """
        super().__init__(params)
        self.sma_short = params['sma_short']
        self.sma_long = params['sma_long']
        self.rsi_period = params['rsi_period']
        self.bb_period = params['bb_period']
        self.bb_std = params['bb_std']
        self.data_window = max(self.sma_long, self.rsi_period, self.bb_period)
        self.historical_data = []

    @classmethod
    def validate_params(cls, params: Dict[str, Any]):
        """
        Valida los parámetros de la estrategia.

        Args:
            params (Dict[str, Any]): Diccionario con los parámetros a validar.

        Raises:
            ValueError: Si los parámetros no son válidos.
        """
        required_params = ['sma_short', 'sma_long', 'rsi_period', 'bb_period', 'bb_std']
        for param in required_params:
            if param not in params:
                raise ValueError(f"Falta el parámetro requerido: {param}")
            if not isinstance(params[param], int) or params[param] <= 0:
                raise ValueError(f"El parámetro {param} debe ser un entero positivo")
        
        if params['sma_short'] >= params['sma_long']:
            raise ValueError("sma_short debe ser menor que sma_long")

    def required_data(self) -> List[str]:
        """
        Especifica los campos de datos requeridos por la estrategia.

        Returns:
            List[str]: Lista de nombres de campos requeridos.
        """
        return ['Close', 'High', 'Low']

    def generate_signal(self, date: pd.Timestamp, data: pd.Series) -> int:
        """
        Genera una señal de trading basada en los datos proporcionados.

        Args:
            date (pd.Timestamp): La fecha para la cual se está generando la señal.
            data (pd.Series): Los datos de mercado para la fecha dada.

        Returns:
            int: La señal generada (1 para compra, -1 para venta, 0 para mantener).
        """
        self.historical_data.append(data)
        if len(self.historical_data) < self.data_window:
            return 0  # No hay suficientes datos para generar una señal

        df = pd.DataFrame(self.historical_data)
    
        # Calcular indicadores
        sma_short = self.calculate_sma(df['Close'], self.sma_short)
        sma_long = self.calculate_sma(df['Close'], self.sma_long)
        rsi = self.calculate_rsi(df['Close'], self.rsi_period)
        upper_bb, lower_bb, middle_bb = self.calculate_bollinger_bands(df['Close'], self.bb_period, self.bb_std)

        # Generar señal
        signal = 0
        current_price = df['Close'].iloc[-1]
    
        # Señal de compra
        if (sma_short.iloc[-1] > sma_long.iloc[-1] and  # Tendencia alcista
            rsi.iloc[-1] < 70 and  # No sobrecomprado
            current_price > middle_bb.iloc[-1]):  # Precio por encima de la media de Bollinger
            signal = 1
    
        # Señal de venta
        elif (sma_short.iloc[-1] < sma_long.iloc[-1] and  # Tendencia bajista
            rsi.iloc[-1] > 30 and  # No sobrevendido
            current_price < middle_bb.iloc[-1]):  # Precio por debajo de la media de Bollinger
            signal = -1

        # Mantener solo los datos necesarios
        self.historical_data = self.historical_data[-self.data_window:]

        return signal

    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, period: int, std: int) -> tuple:
        """Calcula las Bandas de Bollinger."""
        sma = data.rolling(window=period).mean()
        std_dev = data.rolling(window=period).std()
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        return upper_band, lower_band, sma

    @staticmethod
    def calculate_sma(data: pd.Series, period: int) -> pd.Series:
        """Calcula la Media Móvil Simple."""
        return data.rolling(window=period).mean()

    @staticmethod
    def calculate_rsi(data: pd.Series, period: int) -> pd.Series:
        """Calcula el Índice de Fuerza Relativa (RSI)."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

   