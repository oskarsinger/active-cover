import numpy as np

from models.regression import BinaryL2RegularizedBilinearLogisticRegressionModel as BL2RBilinearLRM
from models.regression import BinaryL2RegularizedLogisticRegressionMOdel as BL2RLRM


class ModelTrainer:

    def __init__(self,
        model,
        optimizer
    ):

        self.model = model
        self.optimizer = optimizer


    def get_prediction(h, x):
        
        return self.model.get_prediction(x, h)


    def get_erm(Z):
        
        # TODO: will need to implement weighted logistic etc for this to work
        pass


    def get_error(h, Z):

        X = np.array(
            [x_t for (x_t, _, _) in Z]
        )
        y = np.array(
            [y_t for (_, y_t, _) in Z]
        )
        p_inv = np.array(
            [p_inv_t for (_, _, p_inv_t) in Z]
        )
        y_hat = self.model.get_prediction(X, h)
        incorrect = 1 - np.abs(y - y_hat)

        return np.mean(incorrect * p_inv)
