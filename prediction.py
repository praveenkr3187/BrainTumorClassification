
import torch
import h5py
import numpy as np
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid
from sklearn.preprocessing import StandardScaler

def preprocess(case_path):
    p = h5py.File(case_path,'r')
    for i in p.keys():
        series = np.array(p[i]['image']) 
        original_image= series
        img_2d = original_image.astype(float)
        img_2d_scaled = (np.maximum(img_2d,0) / img_2d.max()) * 255.0
        img_2d_scaled = np.uint8(img_2d_scaled)
        series = img_2d_scaled
        scaler = StandardScaler()
        series = (scaler.fit_transform(series))      
        series = torch.tensor(np.stack((series,)*3, axis=0))
    return series, img_2d_scaled

def predict(image):
    checkpoint_path = './static/myModel'
    model = torch.load(checkpoint_path,map_location=torch.device('cpu'))
    image = torch.unsqueeze(image,dim=0)
    prediction = model(image.float())
    predicted = torch.max(prediction.data,1)[1]
    return predicted.item()


    
