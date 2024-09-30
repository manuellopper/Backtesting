import pandas as pd
import logging

logger = logging.getLogger(__name__)

class Portfolio:
    """
    Representa un portafolio de inversión.

    Esta clase maneja el estado del portafolio, incluyendo el efectivo disponible,
    las tenencias actuales y el valor total del portafolio. Proporciona métodos
    para actualizar el estado del portafolio basado en señales de trading.

    Attributes:
        initial_capital (float): El capital inicial del portafolio.
        cash (float): El efectivo disponible actual.
        holdings (int): El número de acciones actualmente en posesión.
        total_value (float): El valor total actual del portafolio (efectivo + valor de las tenencias).
    """

    def __init__(self, initial_capital: float):
        """
        Inicializa un nuevo objeto Portfolio.

        Args:
            initial_capital (float): El capital inicial para el portafolio.
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.holdings = 0
        self.total_value = initial_capital

    def update(self, date: pd.Timestamp, data: pd.Series, signal: int):
        """
        Actualiza el estado del portafolio basado en la señal de trading recibida.

        Este método ejecuta compras o ventas basadas en la señal recibida y
        actualiza el estado del portafolio en consecuencia.

        Args:
            date (pd.Timestamp): La fecha actual del trading.
            data (pd.Series): Los datos de mercado para la fecha actual.
            signal (int): La señal de trading (-1 para vender, 1 para comprar, 0 para mantener).

        Note:
            Este método asume que 'Close' está presente en el Series de datos proporcionado.
        """
        price = data['Close']
        if signal == 1 and self.cash >= price:  # Buy
            shares_to_buy = self.cash // price
            self.holdings += shares_to_buy
            self.cash -= shares_to_buy * price
            logger.info(
                f"{date}: Compra de {shares_to_buy} acciones a {price}")
        elif signal == -1 and self.holdings > 0:  # Sell
            self.cash += self.holdings * price
            logger.info(f"{date}: Venta de {self.holdings} acciones a {price}")
            self.holdings = 0

        self.total_value = self.cash + self.holdings * price