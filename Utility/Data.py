import os
from random import shuffle 
import dill
import torch
from torch.utils.data import Dataset


def ReadArtistDict(artist_list_path):
    with open(artist_list_path, "r") as fp:
        artist_list = fp.read().splitlines()
    return ArtistListToDict(artist_list)

def ArtistListToDict(artist_list):
    d = {}
    for i in range(len(artist_list)):
        d[artist_list[i]] = i
    return d

def ReadDict(path):
    return dill.load(open(path, "rb"))

def ReadList(path):
    fp = open(path, "r")
    return fp.read().splitlines()

class SetDataset:
    def __init__(self, artist_list, artist_track_dict, track_clip_dict, batch_size, spec_dir_path, slice_start_list, slice_length):
        self.artist_list = artist_list
        self.artist_dict = ArtistListToDict(artist_list)
        self.sub_dataloader = []
        sub_data_loader_params = dict(
            batch_size = batch_size,
            shuffle = True,
            num_workers = 0,
            pin_memory = False
        )

        for artist in artist_list:
            track_list = artist_track_dict[artist]
            clip_list = []
            for track in track_list:
                clip_list += track_clip_dict[artist + "_" + track]
            sub_dataset = SubDataset(artist, clip_list, spec_dir_path, slice_start_list, slice_length)
            self.sub_dataloader.append(torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params))
    
    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.artist_list)


class SubDataset:
    def __init__(self, artist, clip_list, spec_dir_path, slice_start_list, slice_length):
        self.artist = artist
        self.data = []

        for clip_id in clip_list:
            spec_file_path = spec_dir_path + str(clip_id) + ".dat"
            with open(spec_file_path, "rb") as fp:
                loaded_data = dill.load(fp)
            track_name = loaded_data[1]
            artist_name = loaded_data[2]
            clip_spec = loaded_data[0]

            for slice_start in slice_start_list:
                slice_spec = clip_spec[:,slice_start:slice_start+slice_length]
                self.data.append((artist_name, track_name, slice_spec))

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes = 100):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]

class FewShotDataset(Dataset):
    def __init__(self, artist_list, clip_list, spec_dir_path, slice_start_list, slice_length):
        self.artist_dict = ArtistListToDict(artist_list)
        self.data = []

        for clip_id in clip_list:
            spec_file_path = spec_dir_path + str(clip_id) + ".dat"
            with open(spec_file_path, "rb") as fp:
                loaded_data = dill.load(fp)
            track_name = loaded_data[1]
            artist_id = self.artist_dict[loaded_data[2]]
            clip_spec = loaded_data[0]

            for slice_start in slice_start_list:
                slice_spec = clip_spec[:,slice_start:slice_start+slice_length]
                self.data.append((artist_id, track_name, slice_spec))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ArtistIdentificationDataset(Dataset):
    def __init__(self, clip_list_path, spec_dir_path, slice_start_list, slice_length, artist_list_path):
        self.artist_dict = ReadArtistDict(artist_list_path)

        self.data = []
        with open(clip_list_path, "r") as fp:
            clip_list = fp.read().splitlines()

        for clip_id in clip_list:
            spec_file_path = spec_dir_path + str(clip_id) + ".dat"
            with open(spec_file_path, "rb") as fp:
                loaded_data = dill.load(fp)
            track_name = loaded_data[1]
            artist_id = self.artist_dict[loaded_data[2]]
            clip_spec = loaded_data[0]

            for slice_start in slice_start_list:
                slice_spec = clip_spec[:,slice_start:slice_start+slice_length]
                self.data.append((artist_id, track_name, slice_spec))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# [0, 156, 313, 469, 598]