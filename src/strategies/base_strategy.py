from abc import ABC, abstractmethod
from typing import Dict, Any, List
from ..indicators.technical_indicators import create_indicator, TechnicalIndicator

class Strategy(ABC):
    """
    Clase base abstracta para todas las estrategias de trading.

    Esta clase proporciona la estructura básica para implementar estrategias de trading,
    incluyendo la gestión de indicadores técnicos.

    Attributes:
        params (Dict[str, Any]): Diccionario con los parámetros de la estrategia.
        indicators (Dict[str, TechnicalIndicator]): Diccionario de indicadores técnicos.
    """

    def __init__(self, params: Dict[str, Any]):
        """
        Inicializa la estrategia con los parámetros proporcionados.

        Args:
            params (Dict[str, Any]): Diccionario con los parámetros de la estrategia.
        """
        self.validate_params(params)
        self.params = params
        self.indicators = {}

    @classmethod
    @abstractmethod
    def validate_params(cls, params: Dict[str, Any]):
        """
        Valida los parámetros de la estrategia.

        Este método debe ser implementado por cada estrategia específica.

        Args:
            params (Dict[str, Any]): Diccionario con los parámetros a validar.

        Raises:
            NotImplementedError: Si el método no ha sido implementado por la subclase.
        """
        pass

    def add_indicator(self, name: str, indicator_type: str, params: Dict[str, Any]):
        """
        Añade un indicador técnico a la estrategia.

        Args:
            name (str): Nombre para referenciar el indicador.
            indicator_type (str): Tipo de indicador a crear.
            params (Dict[str, Any]): Parámetros para el indicador.
        """
        self.indicators[name] = create_indicator(indicator_type, params)

    def update_indicators(self, data: Dict[str, float]):
        """
        Actualiza todos los indicadores con los nuevos datos.

        Args:
            data (Dict[str, float]): Diccionario con los nuevos datos.
        """
        for indicator in self.indicators.values():
            indicator.update(data)

    @abstractmethod
    def generate_signal(self, date: Any, data: Dict[str, float]) -> int:
        """
        Genera una señal de trading basada en los datos proporcionados.

        Este método debe ser implementado por cada estrategia específica.

        Args:
            date (Any): La fecha para la cual se está generando la señal.
            data (Dict[str, float]): Los datos de mercado para la fecha dada.

        Returns:
            int: La señal generada (1 para compra, -1 para venta, 0 para mantener).

        Raises:
            NotImplementedError: Si el método no ha sido implementado por la subclase.
        """
        pass

    @abstractmethod
    def required_data(self) -> List[str]:
        """
        Especifica los campos de datos requeridos por la estrategia.

        Returns:
            List[str]: Lista de nombres de campos requeridos.
        """
        pass