import numpy as np

class BinomialTree:
    def __init__(
            self,
            S: float,
            K: float,
            T: float,
            r: float,
            sigma: float,
            N: int = 100,
            option_type: str = "call"
    ):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.N = N
        self.option_type = option_type.lower()
        self.dt = T / N
        self.u = np.exp(sigma * np.sqrt(self.dt))
        self.d = 1 / self.u
        self.p = (np.exp(r*self.dt) - self.d) / (self.u - self.d)


    def price(self) -> float:
        ST = np.array([self.S * self.u ** j * self.d ** (self.N - j) for j in range(self.N + 1)])
        if self.option_type == "call":
            option_values = np.maximum(0, ST - self.K)
        elif self.option_type == "put":
            option_values = np.maximum(0, self.K - ST)
        else:
            raise ValueError("option_type invalid: must be 'call' or 'put'")
        
        # backward inductino for calculating discounted payoffs at an expiry node
        # start at expiry time and work backwards, calculating option price at each node
        for i in range(self.N - 1, -1, -1):
            ST = ST[:-1] / self.u
            option_values = np.exp(-self.r * self.dt) * (
                self.p * option_values[1:] + (1 - self.p) * option_values[:-1]
            )
            if self.option_type == 'call':
                option_values = np.maximum(option_values, ST - self.K)
            else:
                option_values = np.maximum(option_values, self.K - ST)

        return option_values[0]

