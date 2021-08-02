import os
from datetime import datetime

import torch

def save_model(model, criterion, optimizer, num_epochs, checkpoint_path='./saved_models'):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    print('Saving model to: ',checkpoint_path)

    checkpoint_file = os.path.join(checkpoint_path, 'model1.ckpt')
            # %(num_epochs+1, datetime.now()))
    print(checkpoint_file)
    # Convert to cpu before saving
    cpu = torch.device('cpu')
    model = model.to(cpu)
    torch.save({
        'model': model.state_dict(),
        'criterion': criterion,
        'optimizer': optimizer,
        'num_epochs': num_epochs
        }, checkpoint_file)

def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)

    model = checkpoint['model']
    criterion = checkpoint['criterion']
    optimizer = checkpoint['optimizer']
    num_epochs = checkpoint['num_epochs']

    return model, criterion, optimizer, \
            num_epochs