from matplotlib import pyplot as plt

from whitehorses.loaders.simple import GaussianLoader
from whitehorses.loaders.supervised import (
    LogisticRegressionLoader,
    BilinearLogisticRegressionLoader
)
from models.regression import BinaryL2RegularizedLogisticRegressionModel as BL2RLRM
from fitterhappier.qn import FullAdaGradOptimizer


# TODO: add weights so it actually tests weighted
# TODO: maybe use this as opportunity to test ModelTrainer class
class WeightedLogisticRegressionTester:

    def __init__(self,
        n,
        p,
        gamma=10**(-2),
        k=None
    ):

        (self.n, self.p, self.k) = (n, p, k)
        self.gamma = gamma
        self.X_loader = GaussianLoader(
            self.n,
            self.p,
            k=self.k
        )
        self.loader = LogisticRegressionGaussianLoader(
            self.X_loader
        )
        self.data = self.loader.get_data()
        self.model = BL2RLRM(
            self.p,
            self.q,
            self.gamma
        )
        self.get_objective = lambda params: self.model.get_objective
            self.data,
            params
        )
        self.get_gradient = lambda params: self.model.get_gradient(
            self.data,
            params
        )
        self.get_projected = lambda params: self.model.get_projected(
            self.data,
            params
        )

    def run(self):
        
        optimizer = FullAdaGradOptimizer(
            self.model.get_parameter_shape(),
            self.get_objective,
            self.get_gradient,
            self.get_projected
        )

        optimizer.run()

        objs = optimizer.objectives

        plt.plot(np.arange(len(objs)), objs)
        plt.show()
