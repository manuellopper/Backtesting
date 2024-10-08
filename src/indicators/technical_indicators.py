from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any

class TechnicalIndicator(ABC):
    """
    Clase base abstracta para todos los indicadores técnicos.
    
    Esta clase proporciona la estructura básica para implementar indicadores técnicos
    que funcionan en modo streaming, actualizándose con cada nuevo dato recibido.

    Attributes:
        params (Dict[str, Any]): Diccionario con los parámetros del indicador.
        current_value (Any): Valor actual del indicador.
    """

    def __init__(self, params: Dict[str, Any]):
        """
        Inicializa el indicador técnico.

        Args:
            params (Dict[str, Any]): Diccionario con los parámetros del indicador.
        """
        self.params = params
        self.current_value = None

    @abstractmethod
    def update(self, data: Dict[str, float]) -> Any:
        """
        Actualiza el indicador con nuevos datos.

        Este método debe ser implementado por cada indicador específico.

        Args:
            data (Dict[str, float]): Diccionario con los nuevos datos.

        Returns:
            Any: El valor actualizado del indicador.
        """
        pass

    def get_value(self) -> Any:
        """
        Devuelve el valor actual del indicador.

        Returns:
            Any: El valor actual del indicador.
        """
        return self.current_value


class SMA(TechnicalIndicator):
    """
    Implementación de la Media Móvil Simple (Simple Moving Average - SMA).

    Esta clase calcula la media móvil simple de un conjunto de datos en modo streaming.

    Attributes:
        period (int): Período para el cálculo de la SMA.
        price_key (str): Clave para acceder al precio en el diccionario de datos.
        buffer (List[float]): Buffer para almacenar los datos históricos necesarios.
    """

    def __init__(self, params: Dict[str, Any]):
        """
        Inicializa el indicador SMA.

        Args:
            params (Dict[str, Any]): Diccionario con los parámetros del indicador.
                Debe incluir 'period' (int) y 'price_key' (str).
        """
        super().__init__(params)
        self.period = params['period']
        self.price_key = params['price_key']
        self.buffer = []

    def update(self, data: Dict[str, float]) -> float:
        """
        Actualiza la SMA con nuevos datos.

        Args:
            data (Dict[str, float]): Diccionario con los nuevos datos.

        Returns:
            float: El valor actualizado de la SMA.
        """
        price = data[self.price_key]
        self.buffer.append(price)
        
        if len(self.buffer) > self.period:
            self.buffer.pop(0)
        
        self.current_value = sum(self.buffer) / len(self.buffer)
        return self.current_value


class RSI(TechnicalIndicator):
    """
    Implementación del Índice de Fuerza Relativa (Relative Strength Index - RSI).

    Esta clase calcula el RSI de un conjunto de datos en modo streaming.

    Attributes:
        period (int): Período para el cálculo del RSI.
        price_key (str): Clave para acceder al precio en el diccionario de datos.
        gains (List[float]): Buffer para almacenar las ganancias.
        losses (List[float]): Buffer para almacenar las pérdidas.
        prev_price (float): Precio anterior para calcular la variación.
    """

    def __init__(self, params: Dict[str, Any]):
        """
        Inicializa el indicador RSI.

        Args:
            params (Dict[str, Any]): Diccionario con los parámetros del indicador.
                Debe incluir 'period' (int) y 'price_key' (str).
        """
        super().__init__(params)
        self.period = params['period']
        self.price_key = params['price_key']
        self.gains = []
        self.losses = []
        self.prev_price = None

    def update(self, data: Dict[str, float]) -> float:
        """
        Actualiza el RSI con nuevos datos.

        Args:
            data (Dict[str, float]): Diccionario con los nuevos datos.

        Returns:
            float: El valor actualizado del RSI.
        """
        price = data[self.price_key]
        
        if self.prev_price is not None:
            change = price - self.prev_price
            gain = max(change, 0)
            loss = max(-change, 0)
            
            self.gains.append(gain)
            self.losses.append(loss)
            
            if len(self.gains) > self.period:
                self.gains.pop(0)
                self.losses.pop(0)
            
            avg_gain = sum(self.gains) / len(self.gains)
            avg_loss = sum(self.losses) / len(self.losses)
            
            if avg_loss == 0:
                self.current_value = 100
            else:
                rs = avg_gain / avg_loss
                self.current_value = 100 - (100 / (1 + rs))
        
        self.prev_price = price
        return self.current_value


class BollingerBands(TechnicalIndicator):
    """
    Implementación de las Bandas de Bollinger.

    Esta clase calcula las Bandas de Bollinger de un conjunto de datos en modo streaming.

    Attributes:
        period (int): Período para el cálculo de las Bandas de Bollinger.
        num_std (float): Número de desviaciones estándar para las bandas superior e inferior.
        price_key (str): Clave para acceder al precio en el diccionario de datos.
        buffer (List[float]): Buffer para almacenar los datos históricos necesarios.
        sma (SMA): Instancia de SMA para calcular la media móvil.
    """

    def __init__(self, params: Dict[str, Any]):
        """
        Inicializa el indicador de Bandas de Bollinger.

        Args:
            params (Dict[str, Any]): Diccionario con los parámetros del indicador.
                Debe incluir 'period' (int), 'num_std' (float) y 'price_key' (str).
        """
        super().__init__(params)
        self.period = params['period']
        self.num_std = params['num_std']
        self.price_key = params['price_key']
        self.buffer = []
        self.sma = SMA({'period': self.period, 'price_key': self.price_key})

    def update(self, data: Dict[str, float]) -> Dict[str, float]:
        """
        Actualiza las Bandas de Bollinger con nuevos datos.

        Args:
            data (Dict[str, float]): Diccionario con los nuevos datos.

        Returns:
            Dict[str, float]: Diccionario con los valores actualizados de las Bandas de Bollinger.
                Incluye 'upper' (banda superior), 'middle' (SMA), y 'lower' (banda inferior).
        """
        price = data[self.price_key]
        self.buffer.append(price)
        
        if len(self.buffer) > self.period:
            self.buffer.pop(0)
        
        middle = self.sma.update(data)
        std = np.std(self.buffer)
        
        upper = middle + (self.num_std * std)
        lower = middle - (self.num_std * std)
        
        self.current_value = {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }
        
        return self.current_value


def create_indicator(indicator_type: str, params: Dict[str, Any]) -> TechnicalIndicator:
    """
    Fábrica de indicadores técnicos.

    Esta función crea y devuelve una instancia del indicador técnico especificado.

    Args:
        indicator_type (str): Tipo de indicador a crear ('SMA', 'RSI', 'BollingerBands').
        params (Dict[str, Any]): Diccionario con los parámetros del indicador.

    Returns:
        TechnicalIndicator: Instancia del indicador técnico especificado.

    Raises:
        ValueError: Si se especifica un tipo de indicador no soportado.
    """
    indicators = {
        'SMA': SMA,
        'RSI': RSI,
        'BollingerBands': BollingerBands
    }
    
    if indicator_type not in indicators:
        raise ValueError(f"Tipo de indicador no soportado: {indicator_type}")
    
    return indicators[indicator_type](params)