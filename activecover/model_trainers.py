import numpy as np

from models.regression import BinaryL2RegularizedBilinearLogisticRegressionModel as BL2RBilinearLRM
from models.regression import BinaryL2RegularizedLogisticRegressionMOdel as BL2RLRM


class ModelTrainer:

    def __init__(self,
        model,
        get_optimizer
    ):

        self.model = model
        self.get_optimizer = get_optimizer


    def get_prediction(h, x):
        
        return self.model.get_prediction(x, h)


    def get_erm(Z):
        
        get_objective = lambda params: self.model.get_objective(
            Z,
            params
        )
        get_gradient = lambda params: self.model.get_gradient(
            Z,
            params
        )
        get_projected = lambda params: self.model.get_projected(
            Z, params
        )
        optimizer = self.get_optimizer(
            self.model.get_parameter_shape(),
            get_objective,
            get_gradient,
            get_projected
        )

        optimizer.run()

        return optimizer.get_parameters()


    def get_error(h, Z):

        y = np.array(
            [z[-2] for z in Z]
        )
        p_inv = np.array(
            [z[-1] for z in Z]
        )
        y_hat = self.model.get_prediction(Z, h)
        incorrect = 1 - np.abs(y - y_hat)

        return np.mean(incorrect * p_inv)
