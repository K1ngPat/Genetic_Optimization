import random as rand

class Connect4():
    
    def __init__(self):
        self.board = [[0 for _ in range(7)] for _ in range(6)]
        self.winner = 0 # 0 for incomplete game, 1 for player 1, 2 for player 2, 3 for tie
        self.moves_played = 0

    def reset(self):
        self.board = [[0 for _ in range(7)] for _ in range(6)]
        self.winner = 0 # 0 for incomplete game, 1 for player 1, 2 for player 2, 3 for tie
        self.moves_played = 0


    def check_this(self, row: int, column: int): #Checks whether specified position is resulting in a win for either side, returns 1 and 2 for respective players, else 0 
        pl_ch = self.board[row][column]
        if pl_ch == 0:
            return -1
        
        # check horizontal
        cnt = 0
        sion = 0

        for i in [1, 2, 3]:
            try:
                sion = self.board[row][column + i]
            except IndexError:
                sion = -1
            
            if sion == pl_ch:
                cnt += 1
            else:
                break

        for i in [1, 2, 3]:
            try:
                sion = self.board[row][column - i]
            except IndexError:
                sion = -1
            
            if sion == pl_ch:
                cnt += 1
            else:
                break

        
        if cnt >= 3:
            return pl_ch
        
        # vertical
        cnt = 0
        sion = 0

        for i in [1, 2, 3]:
            try:
                sion = self.board[row+i][column]
            except IndexError:
                sion = -1
            
            if sion == pl_ch:
                cnt += 1
            else:
                break

        for i in [1, 2, 3]:
            try:
                sion = self.board[row-i][column]
            except IndexError:
                sion = -1
            
            if sion == pl_ch:
                cnt += 1
            else:
                break

        
        if cnt >= 3:
            return pl_ch
        
        # check [primary diagonal
        cnt = 0
        sion = 0

        for i in [1, 2, 3]:
            try:
                sion = self.board[row+i][column + i]
            except IndexError:
                sion = -1
            
            if sion == pl_ch:
                cnt += 1
            else:
                break

        for i in [1, 2, 3]:
            try:
                sion = self.board[row-i][column - i]
            except IndexError:
                sion = -1
            
            if sion == pl_ch:
                cnt += 1
            else:
                break

        
        if cnt >= 3:
            return pl_ch
        

        # check secondary diagonal
        cnt = 0
        sion = 0

        for i in [1, 2, 3]:
            try:
                sion = self.board[row-i][column + i]
            except IndexError:
                sion = -1
            
            if sion == pl_ch:
                cnt += 1
            else:
                break

        for i in [1, 2, 3]:
            try:
                sion = self.board[row+i][column - i]
            except IndexError:
                sion = -1
            
            if sion == pl_ch:
                cnt += 1
            else:
                break

        
        if cnt >= 3:
            return pl_ch
        
        return 0

        
            


    def move(self, column: int, player: int): # column ranges from 0 to 6, player is 1 or 2 
                                # Returns -1 if invalid move, else same scheme as self.winner
        if self.board[0][column] != 0:
            return -1
        
        i=0
        while i<5:
            if self.board[i+1][column] == 0:
                i += 1
            else:
                break

        self.board[i][column] = player
        self.moves_played += 1
        
        t = self.check_this(i, column)
        if t==0 and self.moves_played == 42:
            self.winner = 3
            return 3
        
        self.winner = t
        return t
    
    def play(self, agent1, agent2, turns_played = 0): # turns_played counts number of moves FROM EACH SIDE, eg, 4 moves means 2 turns_played 
        
        
        pat = True
        
        while pat:

            pat = False

            self.reset()

            if turns_played == 0:
                break

            for _ in range(turns_played):
                
                a = self.move(rand.randint(0, 6), 1)
                

                if (a != 0):
                    pat = True
                    break

                a = self.move(rand.randint(0, 6), 2)
                

                if (a != 0):
                    pat = True
                    break

        while True:
            
            a = self.move(agent1.choose_move(self.board, True), 1) # Requirements from agent1 described in arena.py
            if a == -1:
                self.winner = 2
                return 2
            if a != 0:
                self.winner = a
                return a
            
            a = self.move(agent1.choose_move(self.board, False), 2)
            if a == -1:
                self.winner = 1
                return 1
            if a != 0:
                self.winner = a
                return a