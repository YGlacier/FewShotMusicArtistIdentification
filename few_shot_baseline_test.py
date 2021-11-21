import dill
import os
import numpy as np
import torch
from torch.functional import Tensor
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import f1_score

from model import ArtistIdentificationModel, ArtistIdentificationFeatureModel, ArtistIdentificationClassifierModel
from Utility.Data import ArtistIdentificationDataset, ReadArtistDict, ReadList, ReadDict, ArtistListToDict

spec_dir_path = "./Data/spec/"
artist_list_path = "./Data/few_shot_train/few_shot_train_artist_list.txt"
feature_model_weight_path = "./Weights/few_pre/0/0_18_feature.model"
artist_track_dict_path = "./Data/few_shot_train/validation_artist_track_dict.dill"
track_clip_dict_path = "./Data/few_shot_train/validation_track_clip_dict.dill"
feature_seed = 0
classifier_seed = 0
feature_weight_dir = "./Weights/few_pre/" + str(feature_seed) + "/"
classifier_weight_dir = "./Weights/few/" + str(feature_seed) + "/"
slice_length = 313
slice_start_list=[0, 156, 313, 469, 598]

device_id = 0
device = torch.device(("cuda:" + str(device_id)) if torch.cuda.is_available() else "cpu")

k_way = 5
k_shot = 5

artist_track_dict = ReadDict(artist_track_dict_path)
track_clip_dict = ReadDict(track_clip_dict_path)
artist_list = ReadList(artist_list_path)
used_artist_list = artist_list[0:k_way]
used_artist_dict = ArtistListToDict(used_artist_list)
clip_list = []
for artist in used_artist_list:
    track_list = artist_track_dict[artist]
    for track in track_list:
        clip = track_clip_dict[artist + "_" + track]
        clip_list += clip

clip_data = []
for clip_id in clip_list:
    spec_file_path = spec_dir_path + str(clip_id) + ".dat"
    with open(spec_file_path, "rb") as fp:
        loaded_data = dill.load(fp)
        track_name = loaded_data[1]
        artist_id = used_artist_dict[loaded_data[2]]
        clip_spec = loaded_data[0]
    clip_data.append((artist_id, track_name, clip_spec))

feature_model = ArtistIdentificationFeatureModel().to(device)
feature_model.eval()
classifier_model = ArtistIdentificationClassifierModel(k_way).to(device)
classifier_model.eval()
feature_model.load_state_dict(torch.load(feature_weight_dir + str(feature_seed) + "_" + str(22) + "_feature" + ".model"))

accuracy_list = []
for epoch in range(100):
    classifier_model.load_state_dict(torch.load(classifier_weight_dir + str(classifier_seed) + "_" + str(epoch) + "_classifier" + ".model"))

    total_clip = 0.0
    correct_clip = 0.0
    y_true = []
    y_pred = []
    for clip in clip_data:
        clip_spec = clip[2]
        clip_artist = clip[0]

        vote = [0] * k_way
        for slice_start in slice_start_list:
            slice_spec = clip_spec[:,slice_start:slice_start+slice_length]
            input=torch.Tensor(slice_spec).unsqueeze(0).unsqueeze(0).to(device)
            feature = feature_model.forward(input)
            output = classifier_model.forward(feature)
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
    
np.savetxt(classifier_weight_dir + "validation_accuracy.txt", accuracy_list, delimiter="\n")

            

