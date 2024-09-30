import pandas as pd
import numpy as np
import yfinance as yf
from typing import List
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Clase para cargar y manejar datos financieros.

    Esta clase se encarga de descargar datos históricos de precios para un símbolo
    específico utilizando la biblioteca yfinance. Proporciona métodos para cargar
    los datos y obtener un flujo de datos que simula datos en tiempo real.

    Attributes:
        symbol (str): El símbolo del instrumento financiero.
        start_date (str): La fecha de inicio para los datos históricos.
        end_date (str): La fecha de fin para los datos históricos.
        required_fields (List[str]): Los campos de datos requeridos.
        data (pd.DataFrame): Los datos cargados.

    """

    def __init__(self, symbol: str, start_date: str, end_date: str, required_fields: List[str]):
        """
        Inicializa el DataLoader con los parámetros especificados.

        Args:
            symbol (str): El símbolo del instrumento financiero.
            start_date (str): La fecha de inicio para los datos históricos.
            end_date (str): La fecha de fin para los datos históricos.
            required_fields (List[str]): Los campos de datos requeridos.
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.required_fields = required_fields
        self.data = None

    def load_data(self):
        """
        Carga los datos históricos para el símbolo especificado.

        Esta función descarga los datos utilizando yfinance, verifica que todos los
        campos requeridos estén disponibles, y calcula los retornos si 'Close' está
        en los campos requeridos.

        Raises:
            ValueError: Si alguno de los campos requeridos no está disponible en los datos descargados.
        """
        logger.info(
            f"Cargando datos para {self.symbol} desde {self.start_date} hasta {self.end_date}")
        all_data = yf.download(
            self.symbol, start=self.start_date, end=self.end_date)

        # Asegurarse de que todos los campos requeridos estén disponibles
        for field in self.required_fields:
            if field not in all_data.columns:
                raise ValueError(
                    f"El campo requerido '{field}' no está disponible en los datos descargados")

        # Crear una copia explícita de los datos requeridos
        self.data = all_data[self.required_fields].copy()

        if 'Close' in self.required_fields:
            self.data['Returns'] = self.data['Close'].pct_change()

        logger.info(f"Datos cargados: {len(self.data)} filas")

    def get_data_stream(self):
        """
        Genera un flujo de datos que simula datos en tiempo real.

        Yields:
            tuple: Un par (fecha, datos) para cada punto de datos en el conjunto de datos.

        Note:
            Este método es un generador que produce los datos fila por fila,
            simulando un flujo de datos en tiempo real.
        """
        for index, row in self.data.iterrows():
            yield index, row