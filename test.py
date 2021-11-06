import dill
import os
import numpy as np
import torch
from torch.functional import Tensor
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import f1_score

from model import ArtistIdentificationModel
from Utility.Data import ArtistIdentificationDataset, ReadArtistDict

clip_list_path = "./Data/20_artist_identification/validation_clips.txt"
spec_dir_path = "./Data/spec/"
artist_list_path = "./Data/20_artist_identification/20_artist_list.txt"
seed = 1
weight_dir = "./Weights/" + str(seed) + "/"
slice_length = 313
batch_size = 16
slice_start_list=[0, 156, 313, 469, 598]

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

artist_dict = ReadArtistDict(artist_list_path=artist_list_path)
with open(clip_list_path, "r") as fp:
    clip_list = fp.read().splitlines()
clip_data = []

for clip_id in clip_list:
    spec_file_path = spec_dir_path + str(clip_id) + ".dat"
    with open(spec_file_path, "rb") as fp:
        loaded_data = dill.load(fp)
        track_name = loaded_data[1]
        artist_id = artist_dict[loaded_data[2]]
        clip_spec = loaded_data[0]
    clip_data.append((artist_id, track_name, clip_spec))

model = ArtistIdentificationModel().to(device)
model.eval()

accuracy_list = []
for epoch in range(100):
    model.load_state_dict(torch.load(weight_dir + str(seed) + "_" + str(epoch) + ".model"))

    total_clip = 0.0
    correct_clip = 0.0
    y_true = []
    y_pred = []
    for clip in clip_data:
        clip_spec = clip[2]
        clip_artist = clip[0]

        vote = [0] * 20
        for slice_start in slice_start_list:
            slice_spec = clip_spec[:,slice_start:slice_start+slice_length]
            input=torch.Tensor(slice_spec).unsqueeze(0).unsqueeze(0).to(device)
            output = model.forward(input)[0]
            ouput_label = torch.argmax(output)
            vote[ouput_label] += 1
        
        most_vote = torch.argmax(Tensor(vote))

        y_true.append(clip_artist)
        y_pred.append(most_vote)
        
        total_clip += 1
        if most_vote == clip_artist:
            correct_clip += 1
    
    accuracy = correct_clip / total_clip
    accuracy_list.append(accuracy)
    f1 = f1_score(y_true, y_pred, average="weighted")

    print(str(epoch) + "Acc: " + str(accuracy) + ", F1: " + str(f1))
    
np.savetxt(weight_dir + "validation_accuracy.txt", accuracy_list, delimiter="\n")

            

