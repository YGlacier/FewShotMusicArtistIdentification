from numpy.core.fromnumeric import swapaxes
import torch
import sys
from torch.functional import Tensor
import torch.nn.functional as F
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

class RelationNet(FewShotTemplate):
    def __init__(self, device, k_way, k_shot, k_query):
        super(RelationNet, self).__init__(device, k_way, k_shot)
        self.feature_model = ArtistIdentificationFeatureModel().to(self.device)
        self.relation_module = RelationModule(17 * 32 * 2, 8).to(self.device)
        self.loss_function = nn.MSELoss()
        self.k_query = k_query

    def forward_loss(self, x, y):
        feature = self.feature_model(x)
        xf_query = feature[self.k_shot * self.k_way * 5:, :]
        xf_support = feature[:self.k_shot * self.k_way * 5, :]

        y_query = y[self.k_shot * self.k_way * 5:]
        y_support = y[:self.k_shot * self.k_way * 5]

        _, idx = torch.sort(y_support)
        idx = idx.repeat_interleave(544).view(self.k_shot * self.k_way * 5, -1).to(self.device)

        xf_support = torch.gather(xf_support, dim=0, index=idx)
        xf_proto = xf_support.view(self.k_way, self.k_shot*5, -1).mean(1)
        xf_proto_ext = xf_proto.repeat(self.k_query * self.k_way, 1,1)
        xf_query_ext = xf_query.repeat(self.k_way, 1, 1)
        xf_query_ext = torch.transpose(xf_query_ext, 0, 1)
        relation_pairs = torch.cat((xf_proto_ext, xf_query_ext),2).view(-1, 17 * 32 * 2)
        relations = self.relation_module(relation_pairs).view(-1, self.k_way)
        y_oh = one_hot(y_query.type(torch.LongTensor), self.k_way).to(self.device)

        return self.loss_function(relations, y_oh)

    def test_forward(self, x, y):
        feature = self.feature_model(x)
        xf_query = feature[self.k_shot * self.k_way * 5:, :]
        xf_support = feature[:self.k_shot * self.k_way * 5, :]

        y_support = y

        _, idx = torch.sort(y_support)
        idx = idx.repeat_interleave(544).view(self.k_shot * self.k_way * 5, -1).to(self.device)

        xf_support = torch.gather(xf_support, dim=0, index=idx)
        xf_proto = xf_support.view(self.k_way, self.k_shot*5, -1).mean(1)
       
        #xf_proto_ext = xf_proto.repeat(self.k_way, 1, 1)
        xf_proto_ext = xf_proto.unsqueeze(0)
        xf_query_ext = xf_query.repeat(self.k_way, 1, 1)
        xf_query_ext = torch.transpose(xf_query_ext, 0, 1)

        relation_pairs = torch.cat((xf_proto_ext, xf_query_ext),2).view(-1, 17 * 32 * 2)
        relations = self.relation_module(relation_pairs).view(-1, self.k_way)

        return relations


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

            avg_loss += loss.item()

            if not self.is_train_log_muted:
                if i % print_freq == 0:
                    print("Epoch {:d} | Batch {:d}/{:d} | Loss {:f}".format(epoch, i, len(data_loader), avg_loss/float(i+1)))

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


class RelationModule(nn.Module):
    def __init__(self,input_size, hidden_size):
        super(RelationModule, self).__init__()
        self.fc1 = nn.Linear(input_size, int(input_size/2))
        self.bn1 = nn.BatchNorm1d(int(input_size/2))
        input_size = int(input_size/2)
        self.fc2 = nn.Linear(input_size, int(input_size/2))
        self.bn2 = nn.BatchNorm1d(int(input_size/2))
        input_size = int(input_size/2)
        self.fc3 = nn.Linear(input_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)

    def forward(self,x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = F.relu(self.fc3(out))
        out = torch.sigmoid(self.fc4(out))
        return out
