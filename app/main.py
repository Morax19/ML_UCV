import torch
from strokeModel import strokeModelNN

model = strokeModelNN()
model.load_state_dict(torch.load('strokeModelFinal.pth'))