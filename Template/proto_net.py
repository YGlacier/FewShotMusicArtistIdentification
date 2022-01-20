from numpy.core.fromnumeric import swapaxes
import torch
import sys
from torch.functional import Tensor
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn.modules.activation import Softmax
from sklearn.metrics import f1_score
from .few_shot_template import FewShotTemplate
from sklearn import preprocessing
from .model import ArtistIdentificationFeatureModel

def one_hot(y, num_class):         
    return torch.zeros((len(y), num_class)).scatter_(1, y.unsqueeze(1), 1)

class ProtoNet(FewShotTemplate):
    def __init__(self, device, k_way, k_shot, k_query):
        super(ProtoNet, self).__init__(device, k_way, k_shot)
        self.k_query = k_query
        self.feature_model = ArtistIdentificationFeatureModel().to(self.device)
        self.loss_function = nn.CrossEntropyLoss()

    def forward_loss(self, x, y):
        feature = self.feature_model(x)
        xf_query = feature[self.k_shot * self.k_way * 5:, :]
        xf_support = feature[:self.k_shot * self.k_way * 5, :]

        y_query = y[self.k_shot * self.k_way * 5:]
        y_support = y[:self.k_shot * self.k_way * 5]

        _, idx = torch.sort(y_support)
        idx = idx.repeat_interleave(544).view(self.k_shot * self.k_way * 5, -1).to(self.device)

        xf_support = torch.gather(xf_support, dim=0, index=idx)

        xf_support = xf_support.contiguous()
        xf_proto = xf_support.view(self.k_way, self.k_shot * 5, -1).mean(1)
        xf_query = xf_query.contiguous().view(self.k_way * self.k_query, -1)

        dists = euclidean_dist(xf_query, xf_proto)
        y_query = Variable(y_query.type(torch.LongTensor).to(self.device))

        loss = self.loss_function(-dists, y_query)
        del xf_proto, xf_query, xf_support, y_query
        return loss

    def test_forward(self, x, y):
        feature = self.feature_model(x)
        xf_query = feature[self.k_shot * self.k_way * 5:, :]
        xf_support = feature[:self.k_shot * self.k_way * 5, :]

        y_query = y[self.k_shot * self.k_way * 5:]
        y_support = y[:self.k_shot * self.k_way * 5]

        _, idx = torch.sort(y_support)
        idx = idx.repeat_interleave(544).view(self.k_shot * self.k_way * 5, -1)

        xf_support = torch.gather(xf_support, dim=0, index=idx)

        xf_support = xf_support.contiguous()
        xf_proto = xf_support.view(self.k_way, self.k_shot * 5, -1).mean(1)
        xf_query = xf_query.contiguous().view(1, -1)

        dists = euclidean_dist(xf_query, xf_proto)
        scores = -dists
        del xf_proto, xf_query, xf_support, y_query
        return scores

    def train_loop(self, epoch, data_loader, optimizer):
        avg_loss = 0
        print_freq = 10

        for i, data in enumerate(data_loader):
            
            self.label_encoder = preprocessing.LabelEncoder()
            label = torch.Tensor(self.label_encoder.fit_transform(np.array(data[0]).reshape(-1)))

            input_spec = torch.swapaxes(data[2],0,1).reshape(-1, data[2].size(2), data[2].size(3)).unsqueeze(1).to(self.device)

            optimizer.zero_grad()
            loss = self.forward_loss(input_spec, label)
            loss.backward()
            optimizer.step()

            avg_loss += float(loss.item())

            if not self.is_train_log_muted:
                if i % print_freq == 0:
                    print("Epoch {:d} | Batch {:d}/{:d} | Loss {:f}".format(epoch, i, len(data_loader), avg_loss/float(i+1)))
        
        del input_spec, loss
    
    def test_loop(self, support_data, query_data):
        correct = 0
        count = 0
        slice_length = 313
        slice_start_list=[0, 156, 313, 469, 598]
        y_true = []
        y_pred = []

        x_support = []
        y_support = []
        for support_clip in support_data:
            clip_spec = support_clip[2]
            clip_artist = support_clip[0]
            for slice_start in slice_start_list:
                slice_spec = clip_spec[:,slice_start:slice_start+slice_length]
                x_support.append(slice_spec)
                y_support.append(clip_artist)

        for query_clip in query_data:
            clip_spec = query_clip[2]
            clip_artist = query_clip[0]

            vote = [0] * self.k_way
            for slice_start in slice_start_list:
                slice_spec = clip_spec[:,slice_start:slice_start+slice_length]
                query_spec=[slice_spec]
                x_input=torch.Tensor(np.array(x_support + query_spec)).unsqueeze(1).to(self.device)
                y_input=torch.Tensor(np.array(y_support)).to(self.device)

                with torch.no_grad():
                    scores = self.test_forward(x_input, y_input)
                output_label = torch.argmax(scores)
                vote[output_label] += 1
        
            most_vote = int(torch.argmax(Tensor(vote)))
            y_true.append(clip_artist)
            y_pred.append(most_vote)

            count += 1
            if most_vote == clip_artist:
                correct+=1

        accuracy = correct / float(count)
        f1 = f1_score(y_true, y_pred, average="weighted")

        return accuracy, f1

def euclidean_dist(x,y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)