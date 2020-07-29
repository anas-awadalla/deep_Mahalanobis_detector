import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import math
import json
import itertools
import torch
from tqdm import tqdm

def correct_batch(batch):
  while len(batch)<4000:
    batch = np.append(batch,0)
  return batch

class parkinsonsData(Dataset):
    
    def __init__(self, df, col):
        file_df = pd.read_csv("../../../data3/mPower/52461358.csv")
        if col==14:
          files = file_df["deviceMotion_walking_rest.json.items"].tolist()
        elif col==8:
          files = file_df["deviceMotion_walking_outbound.json.items"].tolist()
        else:
          files = file_df["deviceMotion_walking_return.json.items"].tolist()

        self.users = pd.read_csv("../../../data3/mPower/users.csv")
        self.col = col
        self.dataset = []

        for z, a in zip(tqdm(file_df.iterrows()),tqdm(file_df)):
          healthcode = z[1][3]
          try:
            label = self.users.loc[self.users['healthCode'] == healthcode]["professional-diagnosis"]
            if label.values[0]:
                  label=1
            else:
                  label=0
            try:
              if(os.path.exists("../../../data3/mPower/data/"+str(int(z[1][col]))+".json")):
                f = open("../../../data3/mPower/data/"+str(int(z[1][col]))+".json")
                data = json.load(f)
                x=[]
                x.append([])
                x.append([])
                x.append([])
                
                for i in range(0,len(data),2):
                    rot = data[i].get("rotationRate")
                    x[0].append(rot["x"])
                    x[1].append(rot["y"])
                    x[2].append(rot["z"])
                    
                stdev = np.std(np.asarray(x))
                mean = np.mean(np.asarray(x))
                x = ((np.asarray(x)-mean)/stdev)
                                
                x[0] = correct_batch(x[0])
                x[1] = correct_batch(x[1])
                x[2] = correct_batch(x[2])
                
                # stdev = np.std(np.asarray(x))
                # mean = np.mean(np.asarray(x))
                # x = ((np.asarray(x)-mean)/stdev)
                
                self.dataset.append([x,label])
                
            except:
              continue
          except:
            continue

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # dataPt = self.dataset[idx]
        # f = open(dataPt[0])
        # data = json.load(f)
        # x=[]
        # x.append([])
        # x.append([])
        # x.append([])
         
        # for i in range(0,len(data),2):
        #     rot = data[i].get("rotationRate")
        #     x[0].append(rot["x"])
        #     x[1].append(rot["y"])
        #     x[2].append(rot["z"])
                         
        # x[0] = correct_batch(x[0])
        # x[1] = correct_batch(x[1])
        # x[2] = correct_batch(x[2])
        
        # stdev = np.std(np.asarray(x))
        # mean = np.mean(np.asarray(x))
        # x = ((np.asarray(x)-mean)/stdev)
        
        
        return self.dataset[idx]

