import numpy as np
import random as rand

class Arena():
    """
    What we expect of model_class: 
    An __init__ that takes a list of parameters 
    A function that can crossover between 2 different instances of model_class and create a 3rd and 4th

    What we expect of game:
    An __init__ with the option of slightly random beginning positions
    A method play(self, agent1, agent2, standard_start = True) which plays out a game between agent 1, agent 2 and return 1, 2 or 3 if player 1 win, player 2 win or tie respectively
    """

    def __init__(self, model_class, game, model_params = [], initial_pop = 300, max_pop = 500, crossover_rate = 0.5, mutation_rate = 0.05, battle_stochastic = 0.1):
        self.inhabitants = []
        self.game = game
        self.model_class = model_class
        self.initial_pop = initial_pop
        self.model_params = model_params
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.battle_stochastic = battle_stochastic # Chance that the loser in fight_pair survives IMPORTANT: should be less than 0.5

    

    def initialize(self):
        for _ in range(self.initial_pop):
            self.inhabitants.append(self.model_class(self.model_params))
    
    def pop_cutdown(self):
        while True:
            if len(self.inhabitants)<=self.max_pop:
                return

            fight_pair = rand.choices(range(len(self.inhabitants)), k=2) # indices of the 2 inhabitants we make fight
            # TODO: Make them fight, stochastically make stronger survive
            fight_game = self.game()
            res = fight_game.play(self.inhabitants[fight_pair[0]], self.inhabitants[fight_pair[1]])
            brr = rand.random()
            if res == 3:
                if brr>0.5:
                    self.inhabitants.pop(fight_pair[0])
                else:
                    self.inhabitants.pop(fight_pair[1])
            
            ind = (res == 1) ^ (brr<self.battle_stochastic)

            if ind:
                self.inhabitants.pop(fight_pair[1])
                
            else:
                self.inhabitants.pop(fight_pair[0])


            
            
            
        

