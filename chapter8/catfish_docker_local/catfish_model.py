import torch.nn as nn 
from torchvision import models

CatfishClasses = ["cat","fish"]

CatfishModel = models.resnet50()
CatfishModel.fc = nn.Sequential(nn.Linear(CatfishModel.fc.in_features,500),
                  nn.ReLU(),
                  nn.Dropout(), nn.Linear(500,2))
