import random as rand
import tqdm.tqdm as TQDM

class Arena():
    """
    What we expect of model_class: 
    An __init__ that takes a list of parameters 
    A function crossover(self, other_model) that can crossover between 2 different instances of model_class and return a 3rd and 4th 
    A function mutate(self) that can mutate a model and return mutant. Preferably does high probability of small mutations and low probability of huge mutations
        Note: The methods of crossover and mutation, whether discrete or intermediate, and with what values of crossover_d, are considered attributes of the inhabitants, not the environment. hence, they are left out of this class.

    What we expect of game:
    An __init__ with the option of slightly random beginning positions
    A method play(self, agent1, agent2, standard_start = True) which plays out a game between agent 1, agent 2 and return 1, 2 or 3 if player 1 win, player 2 win or tie respectively
    """

    def __init__(self, model_class, game, model_params = [], initial_pop = 300, max_pop = 500, crossover_rate = 0.1, mutation_rate = 0.07, battle_stochastic = 0.1):
        self.inhabitants = []
        self.game = game
        self.model_class = model_class
        self.initial_pop = initial_pop
        self.model_params = model_params # parameters (NOT GENES) for initialization of inhabitants. There must still be some randomness in their initialization
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.battle_stochastic = battle_stochastic # Chance that the loser in fight_pair survives IMPORTANT: should be less than 0.5

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
    
    def reproduction(self): # Takes care of crossovers AND mutations
        for i in range(self.inhabitants):
            for j in range(self.inhabitants):
                if i==j:
                    continue

                tt = rand.random()
                if tt<self.crossover_rate:
                    child1, child2 = self.inhabitants[i].crossover(self.inhabitants[j])
                    self.inhabitants.append(child1)
                    self.inhabitants.append(child2)
        
        for i in range(self.inhabitants):
            tt = rand.random()
            if tt<self.mutation_rate:
                a = self.inhabitants[i].mutate()
                self.inhabitants.append(a)
    
    def train(self, num_gens = 1000):
        
        for i in TQDM(range(num_gens)):
            if i%100 == 99:
                print("Generation %d \n", i+1)
            
            self.reproduction()
            self.pop_cutdown()