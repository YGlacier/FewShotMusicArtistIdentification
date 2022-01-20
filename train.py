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

batch_size = 4

if __name__=="__main__":
    #warnings.filterwarnings("ignore")
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--algorithm", help="algorithm to use", required=True)
    arg_parser.add_argument("--save", help="path to save model weigths", default="", type=str)
    arg_parser.add_argument("--load", help="path to load model weigths", default="", type=str)
    arg_parser.add_argument("--kway", help="k_ways for few-shot learning", default=5, type=int)
    arg_parser.add_argument("--kshot", help="k_shot for few-shot learning", default=5, type=int)
    arg_parser.add_argument("--kquery", help="k_query for few-shot learning", default=1, type=int)
    arg_parser.add_argument("--device", help="device id", default=0, type=int)
    arg_parser.add_argument("--epoch", help="total epoch count", default=100, type=int)
    arg_parser.add_argument("--iter", help="total iterations", default=1, type=int)
    arg_parser.add_argument("--lr", help="learn rate", default=0.001, type=float)
    arg_parser.add_argument("--mute", help="mute log", action="store_true")
    arg_parser.add_argument("--log", help="path to save test results", default="", type=str)
    args = arg_parser.parse_args()
    algorithm = args.algorithm
    k_way = args.kway
    k_shot = args.kshot
    k_query = args.kquery
    device_id = args.device
    total_epoch = args.epoch
    save_path = args.save
    load_path = args.load
    is_muted = args.mute
    iters = args.iter
    lr = args.lr
    log = args.log

    device = device = torch.device("cuda:" + str(device_id) if torch.cuda.is_available() else "cpu")

    # initialize model
    if algorithm == "baseline":
        model = Baseline(device, k_way, k_shot)
        model.load_feature_weight("./Weights/few_pre/0/0_18_feature.model")
        model.set_train()
        optimizer = torch.optim.Adam(model.classifier_model.parameters(), lr=lr)
        #optimizer = torch.optim.SGD(model.classifier_model.parameters(), lr=lr, momentum=0.9, dampening=0.9, weight_decay=0.001)
    elif algorithm == "baseline++":
        model = BaselinePP(device, k_way, k_shot)
        model.load_feature_weight("./Weights/few_prepp/1/1_16_feature.model")
        model.set_train()
        optimizer = torch.optim.Adam(model.classifier_model.parameters(), lr=lr)
        #optimizer = torch.optim.SGD(model.classifier_model.parameters(), lr=lr, momentum=0.9, dampening=0.9, weight_decay=0.001)
    elif algorithm == "matchingnet":
        model = MatchingNet(device, k_way, k_shot, k_query)
        if load_path != "":
            model.load_state_dict(torch.load(load_path))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, dampening=0.9, weight_decay=0.001)
        batch_size = k_shot * 5 + k_query
    elif algorithm == "protonet":
        model = ProtoNet(device, k_way, k_shot, k_query)
        if load_path != "":
            model.load_state_dict(torch.load(load_path))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        #optimizer = torch.optim.SGD(model.feature_model.parameters(), lr=lr, momentum=0.9, dampening=0.9, weight_decay=0.001)
        batch_size = k_shot * 5 + k_query
    elif algorithm == "relationnet":
        model = RelationNet(device, k_way, k_shot, k_query)
        if load_path != "":
            model.load_state_dict(torch.load(load_path))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        #optimizer = torch.optim.SGD(model.feature_model.parameters(), lr=lr, momentum=0.9, dampening=0.9, weight_decay=0.001)
        batch_size = k_shot * 5 + k_query
    else:
        print("Wrong Alogorithm!")
        sys.exit(1)

    if is_muted:
        model.mute_training_log()

    print("Model Initialized")
    

    if algorithm in ["baseline", "baseline++"]:
        artist_track_dict = ReadDict(finetune_artist_track_dict_path)
        track_clip_dict = ReadDict(finetune_track_clip_dict_path)
        artist_list = ReadList(finetune_artist_list_path)

        validation_artist_track_dict = ReadDict(validation_artist_track_dict_path)
        validation_track_clip_dict = ReadDict(validation_track_clip_dict_path)
        validation_artist_list = ReadList(validation_artist_list_path)

        test_artist_track_dict = ReadDict(test_artist_track_dict_path)
        test_track_clip_dict = ReadDict(test_track_clip_dict_path)
        test_artist_list = ReadList(test_artist_list_path)

        acc_list = []
        f1_list = []
        for i in range(iters):
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
                    clip = random.choice(track_clip_dict[artist + "_" + track])
                    used_clip_list.append(clip)

            dataset = FewShotDataset(used_artist_list,
                                    used_clip_list,
                                    spec_dir_path=spec_dir_path,
                                    slice_start_list=[0, 156, 313, 469, 598],
                                    slice_length=313)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            validation_clip_list = []
            for artist in used_artist_list:
                track_list = validation_artist_track_dict[artist]
                for track in track_list:
                    if track in used_track_list:
                        continue
                    clip = validation_track_clip_dict[artist + "_" + track]
                    validation_clip_list += clip

            validation_clip_list = random.sample(validation_clip_list, 50)

            validation_clip_data = []
            for clip_id in validation_clip_list:
                spec_file_path = spec_dir_path + str(clip_id) + ".dat"
                with open(spec_file_path, "rb") as fp:
                    loaded_data = dill.load(fp)
                    track_name = loaded_data[1]
                    artist_id = used_artist_dict[loaded_data[2]]
                    clip_spec = loaded_data[0]
                validation_clip_data.append((artist_id, track_name, clip_spec))

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

            if not is_muted:
                print("Data Loaded. Loaded {:d} slices.".format(len(dataset)))

            # train
            max_validation_acc = 0
            for epoch in range(total_epoch):
                for g in optimizer.param_groups:
                    g["lr"] = lr - epoch * ((lr - 0.00001) / total_epoch)
                model.set_train()
                model.train_loop(epoch, dataloader, optimizer)
                model.set_test()

                
                acc, f1 = model.test_loop(validation_clip_data)
                if acc > max_validation_acc:
                    max_validation_acc = acc
                    if not is_muted:
                        print("Iter {:d}, Epoch {:d}: Update Max Acc: {:f}".format(i, epoch, acc))
                    torch.save(model.classifier_model.state_dict(), save_path)


            # test
            model.set_test()
            model.classifier_model.load_state_dict(torch.load(save_path))

            # test data
            

            acc, f1 = model.test_loop(test_clip_data)
            print("Iter {:d}: Test Acc = {:f}".format(i, acc))
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
            acc_list.append(acc)
            f1_list.append(f1)

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

    elif algorithm in ["matchingnet"]:
        artist_track_dict = ReadDict(train_artist_track_dict_path)
        track_clip_dict = ReadDict(train_track_clip_dict_path)
        artist_list = ReadList(train_artist_list_path)

        support_artist_track_dict = ReadDict(finetune_artist_track_dict_path)
        support_track_clip_dict = ReadDict(finetune_track_clip_dict_path)
        support_artist_list = ReadList(finetune_artist_list_path)

        validation_artist_track_dict = ReadDict(validation_artist_track_dict_path)
        validation_track_clip_dict = ReadDict(validation_track_clip_dict_path)
        validation_artist_list = ReadList(validation_artist_list_path)

        test_artist_track_dict = ReadDict(test_artist_track_dict_path)
        test_track_clip_dict = ReadDict(test_track_clip_dict_path)
        test_artist_list = ReadList(test_artist_list_path)

        # batch_size = k_shot * 5 + k_query

        dataset = SetDataset(artist_list, artist_track_dict, track_clip_dict, batch_size=batch_size, spec_dir_path=spec_dir_path, slice_start_list=[0, 156, 313, 469, 598], slice_length=313)
        sampler = EpisodicBatchSampler(len(dataset), k_way)
        data_loader_param = dict(batch_sampler = sampler, num_workers = 8, pin_memory = True)
        data_loader = DataLoader(dataset, **data_loader_param)

        top_acc = 0.0
        for epoch in range(total_epoch):
            model.train()
            model.train_loop(epoch, data_loader, optimizer)
            model.eval()

            acc_list = []
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
                    track_list = validation_artist_track_dict[artist]
                    for track in track_list:
                        if track in used_track_list:
                            continue
                        clip = validation_track_clip_dict[artist + "_" + track]
                        test_clip_list += clip

                test_clip_list = random.sample(test_clip_list, 50)

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
                if not is_muted:
                    print("Validation Acc: {:f}".format(acc))
            
            acc = np.mean(acc_list)
            print("Epoch {:d}: Validation Acc = {:f}".format(epoch, acc))
            if acc > top_acc:
                top_acc = acc
                print("Top acc updated: {:f}. Save model.".format(top_acc))
                torch.save(model.state_dict(), save_path)
    
    elif algorithm in ["protonet"]:
        artist_track_dict = ReadDict(train_artist_track_dict_path)
        track_clip_dict = ReadDict(train_track_clip_dict_path)
        artist_list = ReadList(train_artist_list_path)

        #artist_track_dict = ReadDict(finetune_artist_track_dict_path)
        #track_clip_dict = ReadDict(finetune_track_clip_dict_path)
        #artist_list = ReadList(finetune_artist_list_path)

        support_artist_track_dict = ReadDict(finetune_artist_track_dict_path)
        support_track_clip_dict = ReadDict(finetune_track_clip_dict_path)
        support_artist_list = ReadList(finetune_artist_list_path)

        validation_artist_track_dict = ReadDict(validation_artist_track_dict_path)
        validation_track_clip_dict = ReadDict(validation_track_clip_dict_path)
        validation_artist_list = ReadList(validation_artist_list_path)

        test_artist_track_dict = ReadDict(test_artist_track_dict_path)
        test_track_clip_dict = ReadDict(test_track_clip_dict_path)
        test_artist_list = ReadList(test_artist_list_path)

        dataset = SetDataset(artist_list, artist_track_dict, track_clip_dict, batch_size=batch_size, spec_dir_path=spec_dir_path, slice_start_list=[0, 156, 313, 469, 598], slice_length=313)
        sampler = EpisodicBatchSampler(len(dataset), k_way)
        data_loader_param = dict(batch_sampler = sampler, num_workers = 8, pin_memory = True)
        data_loader = DataLoader(dataset, **data_loader_param)

        top_acc = 0.0
        for epoch in range(total_epoch):
            model.train()
            model.train_loop(epoch, data_loader, optimizer)
            model.eval()

            acc_list = []
            for i in range(iters):
                # make support data
                used_artist_list = random.sample(support_artist_list, k_way) 
                used_artist_dict = ArtistListToDict(used_artist_list)
                used_clip_list = []
                for artist in used_artist_list:
                    track_list = support_artist_track_dict[artist]
                    used_track_list = random.sample(track_list, k_shot)
                    for track in used_track_list:
                        clip = support_track_clip_dict[artist + "_" + track][0]
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
                    track_list = validation_artist_track_dict[artist]
                    for track in track_list:
                        if track in used_track_list:
                            continue
                        clip = validation_track_clip_dict[artist + "_" + track]
                        test_clip_list += clip

                test_clip_list = random.sample(test_clip_list, 50)

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
                if not is_muted:
                    print("Validation Acc: {:f}".format(acc))
            
            acc = np.mean(acc_list)
            print("Epoch {:d}: Validation Acc = {:f}".format(epoch, acc))
            if acc > top_acc:
                top_acc = acc
                print("Top acc updated: {:f}. Save model.".format(top_acc))
                torch.save(model.state_dict(), save_path)
    
    elif algorithm in ["relationnet"]:
        artist_track_dict = ReadDict(train_artist_track_dict_path)
        track_clip_dict = ReadDict(train_track_clip_dict_path)
        artist_list = ReadList(train_artist_list_path)

        #artist_track_dict = ReadDict(finetune_artist_track_dict_path)
        #track_clip_dict = ReadDict(finetune_track_clip_dict_path)
        #artist_list = ReadList(finetune_artist_list_path)

        support_artist_track_dict = ReadDict(finetune_artist_track_dict_path)
        support_track_clip_dict = ReadDict(finetune_track_clip_dict_path)
        support_artist_list = ReadList(finetune_artist_list_path)

        validation_artist_track_dict = ReadDict(validation_artist_track_dict_path)
        validation_track_clip_dict = ReadDict(validation_track_clip_dict_path)
        validation_artist_list = ReadList(validation_artist_list_path)

        test_artist_track_dict = ReadDict(test_artist_track_dict_path)
        test_track_clip_dict = ReadDict(test_track_clip_dict_path)
        test_artist_list = ReadList(test_artist_list_path)

        dataset = SetDataset(artist_list, artist_track_dict, track_clip_dict, batch_size=batch_size, spec_dir_path=spec_dir_path, slice_start_list=[0, 156, 313, 469, 598], slice_length=313)
        sampler = EpisodicBatchSampler(len(dataset), k_way)
        data_loader_param = dict(batch_sampler = sampler, num_workers = 8, pin_memory = True)
        data_loader = DataLoader(dataset, **data_loader_param)

        top_acc = 0.0
        for epoch in range(total_epoch):
            model.train()
            model.train_loop(epoch, data_loader, optimizer)
            model.eval()

            acc_list = []
            for i in range(iters):
                # make support data
                used_artist_list = random.sample(support_artist_list, k_way) 
                used_artist_dict = ArtistListToDict(used_artist_list)
                used_clip_list = []
                for artist in used_artist_list:
                    track_list = support_artist_track_dict[artist]
                    used_track_list = random.sample(track_list, k_shot)
                    for track in used_track_list:
                        clip = support_track_clip_dict[artist + "_" + track][0]
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
                    track_list = validation_artist_track_dict[artist]
                    for track in track_list:
                        if track in used_track_list:
                            continue
                        clip = validation_track_clip_dict[artist + "_" + track]
                        test_clip_list += clip

                test_clip_list = random.sample(test_clip_list, 50)

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
                if not is_muted:
                    print("Validation Acc: {:f}".format(acc))
            
            acc = np.mean(acc_list)
            print("Epoch {:d}: Validation Acc = {:f}".format(epoch, acc))
            if acc > top_acc:
                top_acc = acc
                print("Top acc updated: {:f}. Save model.".format(top_acc))
                torch.save(model.state_dict(), save_path)
    



    