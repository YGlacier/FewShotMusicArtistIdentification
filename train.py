import torch
import argparse
import sys
import random
import dill
import numpy as np

from torch.utils.data import DataLoader
from Template.baseline import Baseline
from Template.baselinepp import BaselinePP

from Utility.Data import ArtistListToDict, FewShotDataset, ReadArtistDict, ReadList, ReadDict

spec_dir_path = "./Data/spec/"
artist_list_path = "./Data/few_shot_train/few_shot_train_artist_list.txt"
artist_track_dict_path = "./Data/few_shot_train/train_artist_track_dict.dill"
track_clip_dict_path = "./Data/few_shot_train/train_track_clip_dict.dill"
batch_size = 4

if __name__=="__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--algorithm", help="algorithm to use", required=True)
    #arg_parser.add_argument("--save", help="path to save model weigths", required=True)
    arg_parser.add_argument("--kway", help="k_ways for few-shot learning", default=5, type=int)
    arg_parser.add_argument("--kshot", help="k_shot for few-shot learning", default=5, type=int)
    arg_parser.add_argument("--device", help="device id", default=0, type=int)
    arg_parser.add_argument("--epoch", help="total epoch count", default=100, type=int)
    arg_parser.add_argument("--iter", help="total iterations", default=1, type=int)
    arg_parser.add_argument("--mute", help="mute log", action="store_true")
    args = arg_parser.parse_args()
    algorithm = args.algorithm
    k_way = args.kway
    k_shot = args.kshot
    device_id = args.device
    total_epoch = args.epoch
    #save_path = args.save
    is_muted = args.mute
    iter = args.iter

    device = device = torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu")

    # initialize model
    if algorithm == "baseline":
        model = Baseline(device, k_way, k_shot)
        model.load_feature_weight("./Weights/few_pre/0/0_18_feature.model")
        model.set_train()
        optimizer = torch.optim.SGD(model.classifier_model.parameters(), lr=0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)
    elif algorithm == "baseline++":
        model = BaselinePP(device, k_way, k_shot)
        model.load_feature_weight("./Weights/few_prepp/1/1_16_feature.model")
        model.set_train()
        optimizer = torch.optim.SGD(model.classifier_model.parameters(), lr=0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)
    else:
        print("Wrong Alogorithm!")
        sys.exit(1)

    if is_muted:
        model.mute_training_log()

    if not is_muted:
        print("Model Initialized")
    
    # load data
    artist_track_dict = ReadDict(artist_track_dict_path)
    track_clip_dict = ReadDict(track_clip_dict_path)
    artist_list = ReadList(artist_list_path)

    acc_list = []
    f1_list = []
    for i in range(iter):
        if algorithm == "baseline":
            model.reset_classifier_weight()
        elif algorithm == "baseline++":
            model.reset_classifier_weight()
        else:
            pass

        used_artist_list = random.sample(artist_list, k_way) 
        used_artist_dict = ArtistListToDict(used_artist_list)
        used_clip_list = []
        for artist in used_artist_list:
            track_list = artist_track_dict[artist]
            used_track_list = random.sample(track_list, k_shot)
            for track in used_track_list:
                clip = track_clip_dict[artist + "_" + track][0]
                used_clip_list.append(clip)

        dataset = FewShotDataset(used_artist_list,
                                used_clip_list,
                                spec_dir_path=spec_dir_path,
                                slice_start_list=[0, 156, 313, 469, 598],
                                slice_length=313)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        if not is_muted:
            print("Data Loaded. Loaded {:d} slices.".format(len(dataset)))

        # train
        for epoch in range(total_epoch):
            model.train_loop(epoch, dataloader, optimizer)

        # test
        model.set_test()

        # test data
        test_clip_list = []
        for artist in used_artist_list:
            track_list = artist_track_dict[artist]
            for track in track_list:
                if track in used_track_list:
                    continue
                clip = track_clip_dict[artist + "_" + track]
                test_clip_list += clip

        test_clip_data = []
        for clip_id in test_clip_list:
            spec_file_path = spec_dir_path + str(clip_id) + ".dat"
            with open(spec_file_path, "rb") as fp:
                loaded_data = dill.load(fp)
                track_name = loaded_data[1]
                artist_id = used_artist_dict[loaded_data[2]]
                clip_spec = loaded_data[0]
            test_clip_data.append((artist_id, track_name, clip_spec))

        acc, f1 = model.test_loop(test_clip_data)
        acc_list.append(acc)
        f1_list.append(f1)

    acc_list = np.asarray(acc_list)
    f1_list = np.asarray(f1_list)
    acc_mean = np.mean(acc_list)
    acc_std = np.std(acc_list)
    f1_mean = np.mean(f1_list)
    f1_std = np.std(f1_list)

    print("Acc Mean: {:f}, Acc Std: {:f}, F1 Mean: {:f}, F1 Std: {:f}".format(acc_mean, acc_std, f1_mean, f1_std))

    '''
    # save 
    if algorithm == "baseline":
        model.save_classifier_weight(save_path)
    elif algorithm == "baseline++":
        pass
    else:
        pass
    '''


    