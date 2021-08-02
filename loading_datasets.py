import torch

from torchvision import datasets, transforms

batch_size = 16

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

train_set = datasets.MNIST('C:\\Users\\vidya\\Desktop\\Programs\\Python_programs\\TESTING_BENTOML\\Testing_BentoML', train=True, download=True, 
                         transform=transform)

test_set = datasets.MNIST('C:\\Users\\vidya\\Desktop\\Programs\\Python_programs\\TESTING_BENTOML\\Testing_BentoML', train=False, download=True, 
                         transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
