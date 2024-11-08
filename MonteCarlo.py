import numpy as np

class MonteCarloSimulation:
    def __init__(
        self, 
        S: float, 
        T: float, 
        r: float, 
        sigma: float, 
        simulations: int = 10000, 
        steps: int = 100
    ):
        self.S = S
        self.T = T
        self.r = r
        self.sigma = sigma
        self.simulations = simulations
        self.steps = steps

    def simulate(self) -> np.ndarray:
        """Simulate stock price paths using the Monte Carlo method."""
        dt = self.T / self.steps
        price_paths = np.zeros((self.steps + 1, self.simulations))
        price_paths[0] = self.S
        for t in range(1, self.steps + 1):
            z = np.random.standard_normal(self.simulations)
            price_paths[t] = price_paths[t - 1] * np.exp(
                (self.r - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * z
            )
        return price_paths
