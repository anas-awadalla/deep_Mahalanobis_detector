import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import math
import json
import itertools

def correct_batch(batch):
  while len(batch)<4000:
    batch = np.append(batch,0)
  return batch

class parkinsons_data(Dataset):

    def __init__(self, df):
      
        self.dataset = df

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        output = []
        output.append([])
        output.append([])
        output.append([])
        for i in range(0,len(self.dataset[idx]),2):
                x = self.dataset[idx]["x"][i]
                output[0].append(x["x"])
                output[1].append(x["y"])
                output[2].append(x["z"])
            
        stdev = np.std(np.asarray(output))
        mean = np.mean(np.asarray(output))
        output = ((np.asarray(output)-mean)/stdev).tolist()
                        
        output[0] = correct_batch(output[0])
        output[1] = correct_batch(output[1])
        output[2] = correct_batch(output[2])

        return np.asarray(output), self.dataset[idx]["y"]
