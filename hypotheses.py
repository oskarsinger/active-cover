import numpy as np

from drrobert.data_structures import SuffixTree

class BinaryHypothesisDualVariableContainer:

    def __init__(self, data, get_prediction):

        self.data = data
        self.get_prediction = get_prediction
        self.s_tree = SuffixTree() 

    def udpate(self, parameters, dual_var_update):

        hypothesis = [self.get_prediction(x, parameters)
                      for x in self.data]
        (has, value) = self.s_tree.has(hypothesis)
        
        if has:
            value += dual_var_update
        else:
            value = dual_var_update

        self.s_tree.insert(hypothesis, value=value)

    def get_sum(self, index, value):

        # TODO: get dv value of all hypothesis with value at index
        pass
