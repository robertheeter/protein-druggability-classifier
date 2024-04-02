import os
import torch
import numpy as np


class Meter():

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def average(self):
        self.avg = self.sum / self.count
        return self.avg
    

# def normalize(x):
#     return (x - x.min()) / (x.max() - x.min())


# class BestMeter(object):
#     """Computes and stores the best value"""

#     def __init__(self, best_type):
#         self.best_type = best_type  
#         self.count = 0      
#         self.reset()

#     def reset(self):
#         if self.best_type == 'min':
#             self.best = float('inf')
#         else:
#             self.best = -float('inf')

#     def update(self, best):
#         self.best = best
#         self.count = 0

#     def get_best(self):
#         return self.best

#     def counter(self):
#         self.count += 1
#         return self.count
