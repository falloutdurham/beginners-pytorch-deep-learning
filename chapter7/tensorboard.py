import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch import optim
import torch.nn as nn
from torchvision import datasets, transforms,models
import torch.utils.data

# Writer will output to ./runs/ directory by default

writer = SummaryWriter()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
model = torchvision.models.resnet50(False)

model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
images, labels = next(iter(trainloader))

grid = torchvision.utils.make_grid(images)
writer.add_image('images', grid, 0)
writer.add_graph(model, images)

def send_stats(i, module, input, output):
  writer.add_scalar(f"layer {i}-mean",output.data.mean())
  writer.add_scalar(f"layer {i}-stddev",output.data.std())

from functools import partial

for i,m in enumerate(model.children()): 
    m.register_forward_hook(partial(send_stats, i))

# Now train the model and watch output in Tensorboard

optimizer = optim.Adam(model.parameters(), lr=2e-2)
criterion = nn.CrossEntropyLoss()

def train(model, optimizer, loss_fn,  train_loader, val_loader, epochs=20, device='cuda:0'):
    model.to(device)
    for epoch in range(epochs):
        print(f"epoch {epoch}")
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            ww, target = batch
            ww = ww.to(device)
            target= target.to(device)
            output = model(ww)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
        
        model.eval()
        num_correct = 0 
        num_examples = 0
        for batch in val_loader:
            ww, target = batch
            ww = ww.to(device)
            target= target.to(device)
            output = model(ww)
            correct = torch.eq(torch.max(output, dim=1)[1], target).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        print("Epoch {}, accuracy = {:.2f}".format(epoch, num_correct / num_examples))

train(model,optimizer,criterion,train_data_loader,train_data_loader,epochs=5)