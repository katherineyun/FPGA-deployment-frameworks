import torch
from torch import nn
import numpy as np
from model_arch import WaveNet


dtype = torch.float
device = torch.device("cuda")

model = WaveNet()


# q, ap = wavenet(x)
# print(q, ap)
iters = 50

criterion = nn.MSELoss().cuda()

optimizer = torch.optim.SGD(model.parameters(), 0.001,
                                weight_decay=1e-4)


model.train()
train_loss = 0
correct = 0
total = 0
for i in range(iters):
    
    x = np.random.random((8, 1, 1, 1024)).astype('float32')
    images = torch.tensor(x)
    
    x = np.random.random((8,1)).astype('float32')
    labels = torch.tensor(x) 

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs[0], labels)
    loss.backward()
    optimizer.step()
    cur_loss = loss.item()
    print(cur_loss)
    train_loss += cur_loss
    

