import os
import random
from os import listdir

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    normalize])


def create_train_data():
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

        is_cat = 1 if 'cat' in f else 0
        is_dog = 1 if 'dog' in f else 0
        target = [is_cat, is_dog]
        target_list.append(target)
        if len(train_data_list) >= 64:
            train_data.append((torch.stack(train_data_list), target_list))
            train_data_list = []
            target_list = []
            print("Loaded batch {} of {}".format(len(train_data), int(len(listdir("data/catsanddogs/train/")) / 64)))
            print("Percentage done: {}%".format(
                100 * len(train_data) / int(len(listdir("data/catsanddogs/train/")) / 64)))
    return train_data


class Netz(nn.Module):
    def __init__(self):
        super(Netz, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=3)
        self.conv3 = nn.Conv2d(12, 24, kernel_size=3)
        self.conv4 = nn.Conv2d(24, 48, kernel_size=3)
        self.conv5 = nn.Conv2d(48, 96, kernel_size=3)
        self.conv6 = nn.Conv2d(96, 192, kernel_size=3)
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 2)

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
        x = self.conv5(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.dropout1(x)
        x = x.view(-1, 768)
        x = F.relu(self.dropout2(self.fc1(x)))
        x = self.fc2(x)
        return torch.sigmoid(x)


model = Netz()
model.cuda()

optimizer = optim.RMSprop(model.parameters(), lr=1e-3)


def train(epoch, train_data):
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
            epoch, batch_id, len(train_data),
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
    out_item = out.data.max(1, keepdim=True)[1].item()
    print('Cat' if 0 == out_item else 'Dog')
    img.show()
    x = input("")


if __name__ == '__main__':
    if os.path.isfile('models/catsanddogs.pt'):
        model = torch.load('models/catsanddogs.pt')

    train_data = create_train_data()
    for epoch in range(1, 30):
        train(epoch, train_data)
    torch.save(model, 'models/catsanddogs.pt')

    for i in range(1, 30):
        test()
