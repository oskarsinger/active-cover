from whitehorses.loaders.simple import GaussianLoader
from whitehorses.loaders.supervised import (
    LogisticRegressionLoader,
    BilinearLogisticRegressionLoader
)

from activecover.batch import BatchActiveCover
from activecover.trainer  import ModelTrainer


# TODO: initialize model
# TODO: initialize non-active version
# TODO: get both loss sequences and compare plots

class BatchActiveCoverLogisticRegressionTester:

    def __init__(self, 
        n, 
        p, 
        k=None,
        bilinear=False
    ):
        
        (self.n, self.p, self.k) = (n, p, k)
        self.bilinear = bilinear
        self.X_loader = GaussianLoader(
            self.n, 
            self.p, 
            k=self.k
        )

        if self.bilinear:
            self.loader = BilinearLogisticRegressionLoader(
                self.X_loader, self.X_loader
            )
        else:
            self.loader = LogisticRegressionGaussianLoader(
                self.X_loader
            )

        self.trainer = 
        self.bac = BatchActiveCover(
            model_trainer,
            tolerance,
            c1, c2, c3,
            delta,
            gamma,
            alpha, beta, xi
        )

    def run(self):
        pass
