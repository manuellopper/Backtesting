from typing import Dict, Any, List
from .base_strategy import Strategy

class SmaRsiBollingerStrategy(Strategy):
    """
    Estrategia de trading basada en el cruce de Medias Móviles, RSI y Bandas de Bollinger.

    Esta estrategia utiliza tres indicadores técnicos para generar señales de trading:
    - Cruce de Medias Móviles Simples (SMA)
    - Índice de Fuerza Relativa (RSI)
    - Bandas de Bollinger

    Attributes:
        params (Dict[str, Any]): Diccionario con los parámetros de la estrategia.
        indicators (Dict[str, TechnicalIndicator]): Diccionario de indicadores técnicos utilizados.
    """

    def __init__(self, params: Dict[str, Any]):
        """
        Inicializa la estrategia con los parámetros proporcionados.

        Args:
            params (Dict[str, Any]): Diccionario con los parámetros de la estrategia.
        """
        super().__init__(params)
        self.add_indicator('sma_short', 'SMA', {'period': params['sma_short'], 'price_key': 'Close'})
        self.add_indicator('sma_long', 'SMA', {'period': params['sma_long'], 'price_key': 'Close'})
        self.add_indicator('rsi', 'RSI', {'period': params['rsi_period'], 'price_key': 'Close'})
        self.add_indicator('bb', 'BollingerBands', {'period': params['bb_period'], 'num_std': params['bb_std'], 'price_key': 'Close'})

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
            if not isinstance(params[param], (int, float)) or params[param] <= 0:
                raise ValueError(f"El parámetro {param} debe ser un número positivo")
        
        if params['sma_short'] >= params['sma_long']:
            raise ValueError("sma_short debe ser menor que sma_long")

    def required_data(self) -> List[str]:
        """
        Especifica los campos de datos requeridos por la estrategia.

        Returns:
            List[str]: Lista de nombres de campos requeridos.
        """
        return ['Close', 'High', 'Low']

    def generate_signal(self, date: Any, data: Dict[str, float]) -> int:
        """
        Genera una señal de trading basada en los datos proporcionados.

        Args:
            date (Any): La fecha para la cual se está generando la señal.
            data (Dict[str, float]): Los datos de mercado para la fecha dada.

        Returns:
            int: La señal generada (1 para compra, -1 para venta, 0 para mantener).
        """
        self.update_indicators(data)
        
        sma_short = self.indicators['sma_short'].get_value()
        sma_long = self.indicators['sma_long'].get_value()
        rsi = self.indicators['rsi'].get_value()
        bb = self.indicators['bb'].get_value()

        current_price = data['Close']
        
        # Generar señal
        signal = 0
        
        # Señal de compra
        if (sma_short > sma_long and  # Tendencia alcista
            rsi < 70 and  # No sobrecomprado
            current_price > bb['middle']):  # Precio por encima de la media de Bollinger
            signal = 1
        
        # Señal de venta
        elif (sma_short < sma_long and  # Tendencia bajista
            rsi > 30 and  # No sobrevendido
            current_price < bb['middle']):  # Precio por debajo de la media de Bollinger
            signal = -1

        return signal