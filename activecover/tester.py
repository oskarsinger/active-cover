from whitehorses.loaders.simple import GaussianLoader
from whitehorses.loaders.supervised import (
    LogisticRegressionLoader,
    BilinearLogisticRegressionLoader
)


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
                X_loader, X_loader
            )
        else:
            self.loader = LogisticRegressionGaussianLoader(
                X_loader
            )
