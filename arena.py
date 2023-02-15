import numpy as np

class Arena():
    """
    What we expect of model_class: 
    An __init__ that takes a list of parameters 
    A function that can crossover between 2 different instances of model_class and create a 3rd and 4th
    """

    def __init__(model_class, model_params = [], initial_pop = 300, max_pop = 500):
        self.inhabitants = []
        self.model_class = model_class
        self.initial_pop = initial_pop
        self.model_params = model_params

    

    def initialize():
        for _ in range(self.initial_pop):
            self.inhabitants.append(self.model_class(self.model_params))

