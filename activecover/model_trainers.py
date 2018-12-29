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
        
        pass


    def get_error(Z):

        pass
