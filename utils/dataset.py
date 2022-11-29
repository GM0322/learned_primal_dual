import numpy as np
from utils import config
import torch.utils.data as data
import torch
import os
import cupy
from CTOperator import operator

class MedDataset(data.Dataset):
    def __init__(self,path,isGenerate):
        self.args = config.getArgs()
        self.path = path
        self.files = os.listdir(path)
        if isGenerate:
            self.genreate()
        self.proj = torch.zeros(len(self.files),1,self.args.nViews,self.args.nBins)
        self.label = torch.zeros(len(self.files),1,self.args.nSize,self.args.nSize)
        for step, file in enumerate(self.files):
            self.label[step,0,...] = torch.from_numpy(np.fromfile(path + '/' + file, dtype=np.float32)).view(self.args.nSize,self.args.nSize)
            self.proj[step,0,...] = torch.from_numpy(np.fromfile(path + '/../proj/' + file, dtype=np.float32)).view(self.args.nViews,self.args.nBins)

    def __getitem__(self, item):
        return self.proj[item,...],self.label[item,...],self.files[item]

    def __len__(self):
        return len(self.files)

    def genreate(self):
        if(os.path.isdir(self.path+'/../proj') == False):
            os.mkdir(self.path+'/../proj')
        geom = config.getScanParam()
        for file in self.files:
            I = cupy.asarray(np.fromfile(self.path+'/'+file,dtype=np.float32).reshape(geom['nSize'],geom['nSize']))
            p = cupy.zeros((geom['nViews'],geom['nBins']),dtype=cupy.float32)
            operator.fp(I,geom,p)
            p = cupy.asnumpy(p)
            if self.args.isAddNoisy:
                I0 = 5e4
                maxv = p.max()
                counts = I0 * np.exp(-p / maxv)
                noisy_counts = np.random.poisson(counts)
                p = (-np.log(noisy_counts / I0) * maxv).astype(np.float32)
            p.tofile(self.path+'/../proj/'+file)

def dataLoader(path,isGenerate,batchsize,shuffle):
    dataset = MedDataset(path,isGenerate)
    return data.DataLoader(dataset,batch_size=batchsize,shuffle=shuffle)
