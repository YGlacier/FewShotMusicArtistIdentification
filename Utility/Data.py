import os 
import dill
import torch
from torch.utils.data import Dataset


def ReadArtistDict(artist_list_path):
    with open(artist_list_path, "r") as fp:
        artist_list = fp.read().splitlines()
        d = {}
        for i in range(len(artist_list)):
            d[artist_list[i]] = i
    return d


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