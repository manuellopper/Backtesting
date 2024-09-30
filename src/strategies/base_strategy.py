from abc import ABC, abstractmethod
from typing import Dict, Any, List
import pandas as pd

class Strategy(ABC):
    """
    Clase base abstracta para todas las estrategias de inversión.

    Esta clase define la interfaz que todas las estrategias concretas deben implementar.
    Proporciona métodos abstractos para la generación de señales, validación de parámetros
    y especificación de los datos requeridos.

    Attributes:
        params (Dict[str, Any]): Un diccionario que contiene los parámetros de la estrategia.
    """

    def __init__(self, params: Dict[str, Any]):
        """
        Inicializa la estrategia con los parámetros proporcionados.

        Args:
            params (Dict[str, Any]): Un diccionario de parámetros para configurar la estrategia.

        Raises:
            ValueError: Si los parámetros no pasan la validación.
        """
        self.validate_params(params)
        self.params = params

    @abstractmethod
    def generate_signal(self, date: pd.Timestamp, data: pd.Series) -> int:
        """
        Genera una señal de trading basada en los datos proporcionados.

        Este método debe ser implementado por todas las estrategias concretas.

        Args:
            date (pd.Timestamp): La fecha para la cual se está generando la señal.
            data (pd.Series): Los datos de mercado para la fecha dada.

        Returns:
            int: La señal generada. Típicamente:
                 1 para una señal de compra,
                -1 para una señal de venta,
                 0 para mantener la posición actual.
        """
        pass

    @classmethod
    def validate_params(cls, params: Dict[str, Any]):
        """
        Valida los parámetros proporcionados para la estrategia.

        Este método de clase debe ser implementado por todas las estrategias concretas
        para asegurar que los parámetros proporcionados son válidos y suficientes.

        Args:
            params (Dict[str, Any]): Los parámetros a validar.

        Raises:
            ValueError: Si los parámetros no son válidos o están incompletos.
        """
        pass

    @abstractmethod
    def required_data(self) -> List[str]:
        """
        Especifica los campos de datos requeridos por la estrategia.

        Este método debe ser implementado por todas las estrategias concretas
        para indicar qué datos de mercado necesita la estrategia para funcionar.

        Returns:
            List[str]: Una lista de nombres de campos requeridos (por ejemplo, ['Close', 'Volume']).
        """
        pass