import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from Utility.Data import ArtistListToDict, FewShotDataset, ReadArtistDict, ReadList, ReadDict
from model import ArtistIdentificationModel, ArtistIdentificationFeatureModel, ArtistIdentificationClassifierModel

spec_dir_path = "./Data/spec/"
artist_list_path = "./Data/few_shot_train/few_shot_train_artist_list.txt"
feature_model_weight_path = "./Weights/few_pre/0/"
artist_track_dict_path = "./Data/few_shot_train/train_artist_track_dict.dill"
track_clip_dict_path = "./Data/few_shot_train/train_track_clip_dict.dill"
slice_length = 313
batch_size = 16
seed = 0
device_id = 0
learn_rate = 0.0001
total_epoch = 50
weight_dir = "./Weights/few/" + str(seed) + "/"

k_way = 5
k_shot = 5

device = torch.device(("cuda:" + str(device_id)) if torch.cuda.is_available() else "cpu")

artist_track_dict = ReadDict(artist_track_dict_path)
track_clip_dict = ReadDict(track_clip_dict_path)
artist_list = ReadList(artist_list_path)
used_artist_list = artist_list[0:k_way]
used_artist_dict = ArtistListToDict(used_artist_list)
used_clip_list = []
for artist in used_artist_list:
    track_list = artist_track_dict[artist][0:k_shot]
    for track in track_list:
        clip = track_clip_dict[artist + "_" + track][0]
        used_clip_list.append(clip)


dataset = FewShotDataset(used_artist_list,
                         used_clip_list,
                         spec_dir_path=spec_dir_path,
                         slice_start_list=[0, 156, 313, 469, 598],
                         slice_length=313)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print("Data Loaded")

np_random = np.random.RandomState()
np_random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

feature_model = ArtistIdentificationFeatureModel().to(device)
feature_model.load_state_dict(torch.load(feature_model_weight_path))
feature_model.eval()
classifier_model = ArtistIdentificationClassifierModel().to(device)
classifier_model.train()

print("Model Initialized")

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier_model.parameters(), lr=learn_rate)

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
        with torch.no_grad():
            feature = feature_model.forward(input)
        output = classifier_model.forward(feature)
        loss = loss_function(output, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        epoch_loss += loss.item()
        if i % 100 == 99:
            print('Epoch:%d, i:%d loss= %f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

    weight_path = weight_dir + str(seed) + "_" + str(epoch) +  "_classifier" + ".model"
    torch.save(classifier_model.state_dict(), weight_path)
    loss_list.append(epoch_loss)

np.savetxt(weight_dir + "loss.txt", loss_list, delimiter="\n")