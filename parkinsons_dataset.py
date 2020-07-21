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

class testMotionData(Dataset):
    
    def __init__(self, df, users, root_dir = '/home/jupyter/park/', transform=None):
      
        self.dataset = df
        self.root_dir = root_dir
        self.dataArray = []
        self.resultArray = []
        iterData = iter(self.dataset.iterrows())

        k = 0

        for j,z in zip(iterData,tqdm(range(int(len(self.dataset))))):

          j = j[1]
          healthcode = j[3]
        
          label = users.loc[healthcode][0]
          
          for i in [14]:
            if(not math.isnan(j[i])):
                filedir = str(int(j[i]/10000))
                filename = str(j[i])
                length = len(filename)
                filename = filename[0:length-2]

                if(os.path.isfile(self.root_dir+filedir+"/"+filename+".json"))|(os.path.isfile(self.root_dir+"data"+"/"+filename+".json")):
                  if(os.path.isfile(self.root_dir+filedir+"/"+filename+".json")):
                    f = open(self.root_dir+filedir+"/"+filename+".json")
                  else:
                    f = open(self.root_dir+"data/"+filename+".json")
                try:
                    data = json.load(f)
                except:
                    continue
                if data != None:
                    self.dataArray.append([])
                    self.dataArray[k].append([])
                    self.dataArray[k].append([])
                    self.dataArray[k].append([])
                    for i in range(0,len(data),2):
                          x = data[i].get("rotationRate")
                          self.dataArray[k][0].append(x["x"])
                          self.dataArray[k][1].append(x["y"])
                          self.dataArray[k][2].append(x["z"])

                    stdev = np.std(np.asarray(self.dataArray[k]))
                    mean = np.mean(np.asarray(self.dataArray[k]))
                    self.dataArray[k] = ((np.asarray(self.dataArray[k])-mean)/stdev).tolist()

                    self.dataArray[k][0] = correct_batch(self.dataArray[k][0])
                    self.dataArray[k][1] = correct_batch(self.dataArray[k][1])
                    self.dataArray[k][2] = correct_batch(self.dataArray[k][2])

                    if(label):
                      self.resultArray.append(1)
                    else:
                      self.resultArray.append(0)

                    k = k + 1



        self.dataArray = np.asarray(self.dataArray)
        unique, counts = np.unique(np.array(self.resultArray), return_counts=True)
        print(dict(zip(unique, counts)))



    def __len__(self):
        return len(self.resultArray)

    def __getitem__(self, idx):
        return [self.dataArray[idx], self.resultArray[idx]]

