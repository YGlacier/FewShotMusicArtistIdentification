import torch
import argparse
import sys
import random
import dill
import numpy as np
import warnings
from scipy import stats

from torch.utils.data import DataLoader
from Template.baseline import Baseline
from Template.baselinepp import BaselinePP
from Template.matching_net import MatchingNet
from Template.proto_net import ProtoNet
from Template.relation_net import RelationNet

from Utility.Data import *
from sklearn import preprocessing

spec_dir_path = "./Data/spec/"
train_artist_list_path = "./Data/few_shot_pretrain/pretrain_artist_list.txt"
train_artist_track_dict_path = "./Data/few_shot_pretrain/train_artist_track_dict.dill"
train_track_clip_dict_path = "./Data/few_shot_pretrain/train_track_clip_dict.dill"

finetune_artist_list_path = "./Data/few_shot_train/few_shot_train_artist_list.txt"
finetune_artist_track_dict_path = "./Data/few_shot_train/train_artist_track_dict.dill"
finetune_track_clip_dict_path = "./Data/few_shot_train/train_track_clip_dict.dill"

validation_artist_list_path = "./Data/few_shot_train/few_shot_train_artist_list.txt"
validation_artist_track_dict_path = "./Data/few_shot_train/validation_artist_track_dict.dill"
validation_track_clip_dict_path = "./Data/few_shot_train/validation_track_clip_dict.dill"

test_artist_list_path = "./Data/few_shot_train/few_shot_train_artist_list.txt"
test_artist_track_dict_path = "./Data/few_shot_train/test_artist_track_dict.dill"
test_track_clip_dict_path = "./Data/few_shot_train/test_track_clip_dict.dill"


if __name__=="__main__":
    #warnings.filterwarnings("ignore")
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--algorithm", help="algorithm to use", required=True)
    arg_parser.add_argument("--save", help="path to save model weigths", default="", type=str)
    arg_parser.add_argument("--kway", help="k_ways for few-shot learning", default=5, type=int)
    arg_parser.add_argument("--kshot", help="k_shot for few-shot learning", default=5, type=int)
    arg_parser.add_argument("--kquery", help="k_query for few-shot learning", default=1, type=int)
    arg_parser.add_argument("--device", help="device id", default=0, type=int)
    arg_parser.add_argument("--iter", help="total iterations", default=1, type=int)
    arg_parser.add_argument("--mute", help="mute log", action="store_true")
    arg_parser.add_argument("--log", help="path to save test results", default="", type=str)
    args = arg_parser.parse_args()
    algorithm = args.algorithm
    k_way = args.kway
    k_shot = args.kshot
    k_query = args.kquery
    device_id = args.device
    save_path = args.save
    is_muted = args.mute
    iters = args.iter
    log = args.log

    device = device = torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu")

    # initialize model
    if algorithm == "matchingnet":
        model = MatchingNet(device, k_way, k_shot, k_query)
        model.load_state_dict(torch.load(save_path))
        model.eval()
    elif algorithm == "protonet":
        model = ProtoNet(device, k_way, k_shot, k_query)
        model.load_state_dict(torch.load(save_path))
        model.eval()
    elif algorithm == "relationnet":
        model = RelationNet(device, k_way, k_shot, k_query)
        model.load_state_dict(torch.load(save_path))
        model.eval()
    else:
        print("Wrong Alogorithm!")
        sys.exit(1)

    if is_muted:
        model.mute_training_log()

    print("Model Initialized")
    
    if algorithm in ["matchingnet", "protonet", "relationnet"]:
        support_artist_track_dict = ReadDict(finetune_artist_track_dict_path)
        support_track_clip_dict = ReadDict(finetune_track_clip_dict_path)
        support_artist_list = ReadList(finetune_artist_list_path)

        test_artist_track_dict = ReadDict(test_artist_track_dict_path)
        test_track_clip_dict = ReadDict(test_track_clip_dict_path)
        test_artist_list = ReadList(test_artist_list_path)
            
        acc_list = []
        f1_list = []
        for i in range(iters):
            # make support data
            used_artist_list = random.sample(support_artist_list, k_way) 
            used_artist_dict = ArtistListToDict(used_artist_list)
            used_clip_list = []
            for artist in used_artist_list:
                track_list = support_artist_track_dict[artist]
                used_track_list = random.sample(track_list, k_shot)
                for track in used_track_list:
                    clip = random.choice(support_track_clip_dict[artist + "_" + track])
                    used_clip_list.append(clip)

            support_clip_data = []
            for clip_id in used_clip_list:
                spec_file_path = spec_dir_path + str(clip_id) + ".dat"
                with open(spec_file_path, "rb") as fp:
                    loaded_data = dill.load(fp)
                    track_name = loaded_data[1]
                    artist_id = used_artist_dict[loaded_data[2]]
                    clip_spec = loaded_data[0]
                support_clip_data.append((artist_id, track_name, clip_spec))

            # make test data
            test_clip_list = []
            for artist in used_artist_list:
                track_list = test_artist_track_dict[artist]
                for track in track_list:
                    if track in used_track_list:
                        continue
                    clip = test_track_clip_dict[artist + "_" + track]
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

            acc, f1 = model.test_loop(support_clip_data, test_clip_data)
            acc_list.append(acc)
            f1_list.append(f1)
            
            print("Iter {:d} Test Acc: {:f}".format(i, acc))
            if acc < 0.50:
                print("-------------------")
                print("Bad Acc!")
                print("Artists:")
                for artist in used_artist_list:
                    print(artist)
                print("Support Clips:")
                for clip_id in used_clip_list:
                    print(clip_id)
                print("-------------------")
            
        acc_list = np.asarray(acc_list)
        f1_list = np.asarray(f1_list)
        acc_mean = np.mean(acc_list)
        acc_std = np.std(acc_list)
        f1_mean = np.mean(f1_list)
        f1_std = np.std(f1_list)

        acc_interval_bottom, acc_interval_up = stats.t.interval(alpha = 0.95, df = len(acc_list)-1, loc = acc_mean, scale=stats.sem(acc_list))
        f1_interval_bottom, f1_interval_up = stats.t.interval(alpha = 0.95, df = len(f1_list)-1, loc = f1_mean, scale=stats.sem(f1_list))
        f1_interval = (f1_interval_up - f1_interval_bottom) / 2
        acc_interval = (acc_interval_up - acc_interval_bottom) / 2


        print("Acc Mean: {:f}, Acc Std: {:f}, Acc Interval: {:f}, F1 Mean: {:f}, F1 Std: {:f}, F1 Interval: {:f}".format(acc_mean, acc_std, acc_interval, f1_mean, f1_std, f1_interval))
        np.savetxt(log, acc_list)
    else:
        pass






    '''
    # save 
    if algorithm == "baseline":
        model.save_classifier_weight(save_path)
    elif algorithm == "baseline++":
        pass
    else:
        pass
    '''


    