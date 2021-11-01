import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from Utility.Data import ArtistIdentificationDataset, ReadArtistDict
from model import ArtistIdentificationModel

clip_list_path = "./Data/20_artist_identification/train_clips.txt"
spec_dir_path = "./Data/spec/"
artist_list_path = "./Data/20_artist_identification/20_artist_list.txt"
weight_dir = "./Weights/1/"
slice_length = 313
batch_size = 16
seed = 1
learn_rate = 0.0001
total_epoch = 100

artist_dict = ReadArtistDict(artist_list_path)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

dataset = ArtistIdentificationDataset(clip_list_path=clip_list_path,
                                      spec_dir_path=spec_dir_path,
                                      slice_start_list=[0, 156, 313, 469, 598],
                                      slice_length=313,
                                      artist_list_path=artist_list_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print("Data Loaded")

np_random = np.random.RandomState()
np_random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


model = ArtistIdentificationModel().to(device)
model.train()

print("Model Initialized")

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

loss_list = []
for epoch in range(total_epoch):
    epoch_loss = 0.0
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        input = data[2].unsqueeze(1).to(device)
        '''
        label = list(data[0])

        for j in range(len(label)):
            label[j] = artist_dict[label[j]]
        label = torch.LongTensor(label).to(device)
        '''
        label = data[0].to(device)

        optimizer.zero_grad()
        output = model.forward(input)
        loss = loss_function(output, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        epoch_loss += loss.item()
        if i % 100 == 99:
            print('Epoch:%d, i:%d loss= %f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

    weight_path = weight_dir + str(seed) + "_" + str(epoch) + ".model"
    torch.save(model.state_dict(), weight_path)
    loss_list.append(epoch_loss)

np.savetxt(weight_dir + "loss.txt", loss_list, delimiter="\n")