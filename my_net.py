import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset

train_data = []
train_labels = []
epoch = 50

class My_Network(nn.Module):
    def __init__(self,input_dim = 784, hidden_dim=128, output_dim = 10):
        super(My_Network,self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim,hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim,output_dim)

    def forward(self,x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class CustomDataset(Dataset):
    def __init__(self,data,labels,tranform=None):
        self.data = torch.tensor(data,dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = tranform

    def __len__(self):
        return self.data

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label

train_dataset = CustomDataset(train_data,train_labels)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

#开始初始化
is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")
model = My_Network().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)