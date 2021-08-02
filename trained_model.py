import torch
import torch.nn as nn
import torch.optim as optim

from loading_datasets import train_loader, test_loader, transforms, batch_size
import numpy as np
import matplotlib.pyplot as plt
from tqdm import notebook
import copy
# -------------------------------------------------------------------------------------------------------------------------------
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
class mnist_pred(nn.Module):
    def __init__(self):
        super(mnist_pred, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(784, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# -------------------------------------------------------------------------------------------------------------------------------

def evaluation(dataloader, model):
  total, correct = 0, 0
  for data in dataloader:
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    _, pred = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (pred == labels).sum().item()
  return 100 * correct / total

# -------------------------------------------------------------------------------------------------------------------------------

def train(model, loss_fn, opt, trainloader):
  loss_epoch_arr = []
  max_epochs = 1
  min_loss = 1000

  n_iters = np.ceil(50000/batch_size)
  for epoch in notebook.tqdm(range(max_epochs), total=max_epochs, unit="epochs"):
    for i, data in enumerate(trainloader, 0):
      inputs, labels = data
      inputs, labels = inputs.to(device), labels.to(device)
      opt.zero_grad()
      outputs = model(inputs)
      loss = loss_fn(outputs, labels)
      loss.backward()
      opt.step()

      if min_loss > loss.item():
        min_loss = loss.item()
        best_model = copy.deepcopy(model.state_dict())
        print("Min loss %0.2f" %min_loss)
      del inputs, labels, outputs
      if torch.cuda.is_available():
        torch.cuda.empty_cache()
      if i % 2000 == 0:
            print('Iteration: %d/%d, Loss: %0.2f' % (i, n_iters, loss.item()))
        
    loss_epoch_arr.append(loss.item())
        
    print('Epoch: %d/%d, Test acc: %0.2f, Train acc: %0.2f' % (
        epoch, max_epochs, 
        evaluation(test_loader, model), evaluation(trainloader, model)))
    
  plt.plot(loss_epoch_arr)
  plt.show()
  return best_model    

# -------------------------------------------------------------------------------------------------------------------------------
model_v1 = mnist_pred()
model_v1 = model_v1.to(device)
loss_fn = nn.CrossEntropyLoss()
opt = optim.SGD(model_v1.parameters(), lr=0.05)
# -------------------------------------------------------------------------------------------------------------------------------

best_model = train(model_v1, loss_fn, opt, train_loader)
model_v1.load_state_dict(best_model)
print(evaluation(train_loader, model_v1), evaluation(test_loader, model_v1))

import utils1

utils1.save_model(model_v1, loss_fn, opt, 2)
