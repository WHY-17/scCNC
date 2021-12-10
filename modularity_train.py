import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
from evaluation import eva
from helper import split_train_val_byclass
from modularity import CapsuleNetwork
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import h5py
from sklearn.mixture import GaussianMixture
from collections import Counter
from preprocessing import preprocessing

class MyDataset(Dataset):
    """Operations with the datasets."""

    # def __init__(self, train_data, train_labels, fig_h, fig_w, transform=None):
    def __init__(self, train_data, train_labels, fig_h, fig_w, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # self.data = pd.read_csv(d_file, index_col=0)
        # d = pd.read_csv(cls_file, index_col=0)  #
        self.data = pd.DataFrame(train_data.T)
        d = pd.DataFrame(train_labels)
        self.data_cls = pd.Categorical(d.iloc[:, 0]).codes  #
        self.transform = transform
        self.fig_h = fig_h  ##
        self.fig_w = fig_w

    def __len__(self):
        return len(self.data_cls)

    def __getitem__(self, idx):
        # use astype('double/float') to sovle the runtime error caused by data mismatch.
        data = self.data.iloc[:, idx].values[0:(self.fig_w * self.fig_h), ].reshape(self.fig_w, self.fig_h, 1).astype(
            'double')  #
        label = np.array(self.data_cls[idx]).astype('int32')  #
        sample = {'data': data, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data,label = sample['data'], sample['label']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        data = data.transpose((2, 0, 1))

        return {'data': torch.from_numpy(data),
                'label': torch.from_numpy(label)
                }


def Construct_all_data(loader):
    x = np.array(loader.dataset.data).T.astype(np.float32)
    x = x.reshape(len(x),1,50,40)
    y = loader.dataset.data_cls
    return x, y


Tensor = torch.FloatTensor
f = h5py.File('kolod.h5')
count = np.array(f['X']).astype(np.float32)
all_data = preprocessing(count)
labels = list(f['Y'])

train_data, train_labels, val_data, val_labels = split_train_val_byclass(all_data,labels,0.8)
fig_h = 40
fig_w = 50
learning_rate = 0.001
# Stop training if loss goes below this threshold.
early_stop_loss = 0.0001
batch_size = 256
pre_batch_size = 64
# Configure data loader
transformed_dataset = MyDataset(train_data=train_data,
                                train_labels=train_labels,
                                fig_h=fig_h,
                                fig_w=fig_w,
                                transform=transforms.Compose([
                                    #                                               Rescale(256),
                                    #                                               RandomCrop(224),
                                    ToTensor()
                                ]))
train_loader = DataLoader(transformed_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0, drop_last=True)
# true_label = train_loader.dataset.data_cls
val_transformed_dataset = MyDataset(train_data=val_data,
                                    train_labels=val_labels,
                                    fig_h=fig_h,
                                    fig_w=fig_w,
                                    transform=transforms.Compose([
                                        ToTensor()
                                    ]))
val_loader = DataLoader(val_transformed_dataset, batch_size=pre_batch_size,
                        shuffle=True, num_workers=0, drop_last=True)



conv_inputs = 1
conv_outputs = 256
num_primary_units = 8
primary_unit_size = 32 * 13 * 9 # fixme get from conv2d
# primary_unit_size = 32 * 6 * 6
output_unit_size = 4
num_output_units = 10
n_cluster = num_output_units
n_z = num_output_units * output_unit_size

network = CapsuleNetwork(image_width=fig_h,
                         image_height=fig_w,
                         image_channels=1,
                         conv_inputs=conv_inputs,
                         conv_outputs=conv_outputs,
                         num_primary_units=num_primary_units,
                         primary_unit_size=primary_unit_size,
                         num_output_units=num_output_units, # one for each MNIST digit
                         output_unit_size=output_unit_size,
                         n_clusters=n_cluster,
                         n_z=n_z)
print(network)


# Converts batches of class indices to classes of one-hot vectors.
def to_one_hot(x, length):
    batch_size = x.size(0)
    x_one_hot = torch.zeros(batch_size, length)
    for i in range(batch_size):
        x_one_hot[i, x[i]] = 1.0
    return x_one_hot


def train(epoch):
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    last_loss = None
    log_interval = 1
    for batch_idx, batch_sample in enumerate(train_loader):
        for val_idx, val_sample in enumerate(val_loader):
            val_data = val_sample['data'].type(Tensor)
            val_target = val_sample['label']
            val_target_one_hot = to_one_hot(val_target, length=network.digits.num_units)
            val_data, val_target = Variable(val_data), Variable(val_target_one_hot)
            optimizer.zero_grad()
            val_output = network(val_data)
            val_loss = network.loss(val_data, val_output, val_target)
            val_loss.backward()
            optimizer.step()
            if val_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    val_idx * len(val_data),
                    len(val_loader.dataset),
                    100. * val_idx / len(val_loader),
                    # loss.data[0]))
                    val_loss.item()))
            val_loss = val_loss.item()
            data = batch_sample['data'].type(Tensor)
            target = batch_sample['label']
            target_one_hot = to_one_hot(target, length=network.digits.num_units)
            data, target = Variable(data), Variable(target_one_hot)
            optimizer.zero_grad()
            output = network(data)
            loss = network.reconstruction_loss(data, output) + val_loss
            loss.backward()
            # last_loss = loss.data[0]
            last_loss = loss.item()
            optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                # loss.data[0]))
                loss.item()))
        if last_loss < early_stop_loss:
            break


    return last_loss


num_epochs = 100
for epoch in range(1, num_epochs + 1):
    last_loss = train(epoch)


x,y = Construct_all_data(train_loader)
digit = network.forward(torch.tensor(x))
digit = digit.reshape(len(train_loader.dataset),n_z).detach().numpy()
GM = GaussianMixture(n_components=3)
pred_label = GM.fit_predict(digit)
acc, nmi, ari = eva(y, pred_label)
