import numpy as np
from scipy.stats import norm

class BlackScholes:
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float) -> None:
        """_summary_

        Args:
            S (float): Current Asset Price
            K (float): Strike Price
            T (float): Time To Maturity (in y)
            r (float): Risk-free Interest Rate
            sigma (float): Volatility of asset
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

        self._calculate_d1_d2()

    def _calculate_d1_d2(self) -> None:
        # d1 = number of std devs which rexpected return of asset exceeds strike price
        self.d1 = (np.log(self.S/self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma*np.sqrt(self.T))
        # d2 adjusts for the volatility over the time to maturity
        self.d2 = self.d1 - self.sigma * np.sqrt(self.T)

    def call_price(self) -> float:
        return self.S * norm.cdf(self.d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
    
    def put_price(self) -> float:
        return self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) - self.S * norm.cdf(-self.d1)
    
    def delta(self, option_type: str = "call") -> float:
        # measure of change in option price 
        if option_type == "call":
            return norm.cdf(self.d1)
        elif option_type == "put":
            return norm.cdf(self.d1) - 1
        else:
            raise ValueError("option_type invalid: must be 'call' or 'put'")
        
    def gamma(self) -> float:
        # measures delta's rate of change over time as well as rate of change in underlying asset
        # ehlpful for forecasting price movement
        return norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))
    
    def vega(self) -> float:
        # measures risk of changes in implied volatility
        return self.S * norm.pdf(self.d1) * np.sqrt(self.T)
    
    def theta(self, option_type: str = "call") -> float:
        # measures time decay in the value of an option
        t1 = - (self.S * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T))
        if option_type == "call":
            t2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
            return t1 - t2
        elif option_type == "put":
            t2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
            return t1 + t2
        else:
            raise ValueError("option_type invalid: must be 'call' or 'put'")
        
    def rho(self, option_type: str = "call") -> float:
        # change in price for a derivative relative to change in RF rate of interest
        if option_type == "call":
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        elif option_type == 'put':
            return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        