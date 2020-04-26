import random
from os import listdir

import torch
import torch.optim as optim
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    normalize])

# Target: ([isCat, isDog]
train_data_list = []
target_list = []
train_data = []
files = listdir("data/catsanddogs/train/")
for i in range(len(listdir("data/catsanddogs/train/"))):
    f = random.choice(files)
    files.remove(f)
    img = Image.open("data/catsanddogs/train/" + f)
    img_tensor = transforms(img)
    train_data_list.append(img_tensor)

    isCat = 1 if 'cat' in f else 0
    isDog = 1 if 'dog' in f else 0
    target = [isCat, isDog]
    target_list.append(target)
    if len(train_data_list) >= 64:
        train_data.append((torch.stack(train_data_list), target_list))
        train_data_list = []
        target_list = []
        print("Loaded batch {} of {}".format(len(train_data), int(len(listdir("data/catsanddogs/train/")) / 64)))
        print("Percentage done: {}%".format(100 * len(train_data) / int(len(listdir("data/catsanddogs/train/")) / 64)))
        if len(train_data) > 80:
            break


class Netz(nn.Module):
    def __init__(self):
        super(Netz, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=5)
        self.conv3 = nn.Conv2d(12, 18, kernel_size=5)
        self.conv4 = nn.Conv2d(18, 24, kernel_size=5)
        self.fc1 = nn.Linear(3456, 1000)
        self.fc2 = nn.Linear(1000, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = x.view(-1, 3456)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)


model = Netz()
model.cuda()


optimizer = optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()
    batch_id = 0
    for data, target in train_data:
        data = data.cuda()
        target = torch.Tensor(target).cuda()
        data = Variable(data)
        target = Variable(target)
        optimizer.zero_grad()
        out = model(data)
        criterion = F.binary_cross_entropy
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_id * len(train_data), len(train_data),
                   100. * batch_id / len(train_data), loss.item()))
        batch_id = batch_id + 1


def test():
    model.eval()
    files = listdir("data/catsanddogs/test/")
    f = random.choice(files)
    img = Image.open("data/catsanddogs/test/" + f)
    img_eval_tensor = transforms(img)
    img_eval_tensor.unsqueeze_(0)
    data = Variable(img_eval_tensor.cuda())
    out = model(data)
    print(out.data.max(1, keepdim=True).indices[0])
    img.show()
    x = input("")


for epoch in range(1, 30):
    train(epoch)
test()
