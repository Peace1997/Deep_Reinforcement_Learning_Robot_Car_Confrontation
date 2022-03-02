import math
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from agent.move import NaiveMove

class redAgent():
    def __init__(self):
        # state
        self.health = 20
        self.pos = [0,0]
        self.angle = 0
        self.detect = False
        self.scan = [0]*120
        # action
        self.v_t = 0.0 
        self.v_n = 0.0
        self.angular = 0.0
        self.shoot = 0.0
    def decode_action(self,action:float):
        self.v_t = action[0]
        self.v_n = action[1]
        self.angular = action[2]
        if self.detect == True:
            self.shoot = True


class blueAgent():
    def __init__(self):
        self.avaiable_pos = [
            [0.5, 0.5], [0.5, 2.0], [0.5, 3.0], [0.5, 4.5], # 0 1 2 3 
            [1.5, 0.5], [1.5, 3.0], [1.5, 4.5],             # 4 5 6
            [2.75, 0.5], [2.75, 2.0], [2.75, 3.0], [2.75, 4.5], # 7 8 9 10
            [4.0, 1.75], [4.0, 3.25],                         # 11 12
            [5.25, 0.5], [5.25, 2.0], [5.25, 3.0], [5.25, 4.5], # 13 14 15 16
            [6.5, 0.5], [6.5, 2.0], [6.5, 4.5],             # 17 18 19
            [7.5, 0.5], [7.5, 2.0], [7.5, 3.0], [7.5, 4.5]  # 20 21 22 23
        ]
        self.connected = [
            [1,2,3,4], [0,2,3], [0,1,3,5], [0,1,2,6],
            [0,7], [2,9], [3,10],
            [8,9,10,4], [7,9,10,11], [7,8,10,5,12], [7,8,9],
            [8,14], [9, 15],
            [14,15,16,17], [13,15,16,18,11,11,11,11,11], [13,14,16,12,12,12,12,12], [13,14,15,19],
            [13,20], [14,21], [16, 23],
            [21,22,23,17], [20,22,23,18], [20,21,23], [20,21,22,19]
        ]
        self.path = [
            [5.0, 4.5],
            [5.0, 3.0],
            [4.0, 3.0],
            [3.5, 4.7],
            [0.5, 4.5],
            [0.5, 3.0],
            [2.5, 3.0],
            [4.0, 1.5],
            [7.5, 2.0],
            [7.5, 4.5],
        ]
        self.index = len(self.avaiable_pos)-1
        #self.target = self.path[self.index]
        self.index = random.choice(self.connected[self.index])
        self.target = self.avaiable_pos[self.index]
        self.move = NaiveMove()
        # state
        self.health = 20
        self.pos = [0,0]
        self.angle = 0
        self.detect = False
        self.scan = [0]*120
        # action
        self.v_t = 0.0 
        self.v_n = 0.0
        self.angular = 0.0
        self.shoot = 0.0

    def reset(self, pos):
        #self.index = len(self.avaiable_pos)-1
        self.index = self.avaiable_pos.index(pos)
        self.index = random.choice(self.connected[self.index])
        self.target = self.avaiable_pos[self.index]
        pass

    def decode_action(self,action:float):
        self.v_t = action[0]
        self.v_n = action[1]
        self.angular = action[2]
        if self.detect == True:
            self.shoot = True

    # def decode_action(self,action:float):
    #     self.v_t = action[0]
    #     self.v_n = action[1]
    #     self.angular = action[2]
    #     self.shoot = action[3]
