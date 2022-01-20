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

class MatchingNet(FewShotTemplate):
    def __init__(self, device, k_way, k_shot, k_query):
        super(MatchingNet, self).__init__(device, k_way, k_shot)
        self.k_query = k_query
        self.feature_model = ArtistIdentificationFeatureModel().to(self.device)
        self.loss_function = nn.NLLLoss()
        self.FCE = FullyContextualEmbedding(17 * 32).to(self.device)
        self.G_encoder = nn.LSTM(17 * 32, 17 * 32, 1, batch_first = True, bidirectional=True).to(self.device)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()


    # x = [k_shot * k_way個のサポート特徴] + [k_query * k_way個のクエリ特徴]
    # yも同様
    def forward_loss(self, x, y):
        feature = self.feature_model(x)
        xf_query = feature[self.k_shot * self.k_way * 5:, :]
        xf_support = feature[:self.k_shot * self.k_way * 5, :]

        y_query = y[self.k_shot * self.k_way * 5:]
        y_support = y[:self.k_shot * self.k_way * 5]

        xf_support = xf_support.contiguous().view(self.k_way * self.k_shot * 5, -1)
        xf_query = xf_query.contiguous().view(self.k_way * self.k_query, -1)
        G, G_normalized = self.encode_training_set(xf_support)

        Y_S = Variable(one_hot(y_support.type(torch.LongTensor), self.k_way)).to(self.device)
        f = xf_query

        logprobs = self.get_logprobs(f, G, G_normalized, Y_S)

        y_query = Variable(y_query.type(torch.LongTensor).to(self.device))

        loss = self.loss_function(logprobs, y_query)
        return loss

    def test_forward(self, x, y):
        feature = self.feature_model(x)
        xf_query = feature[self.k_shot * self.k_way * 5:, :]
        xf_support = feature[:self.k_shot * self.k_way * 5, :]
        
        y_support = y

        xf_support = xf_support.contiguous().view(self.k_way * self.k_shot * 5, -1)
        xf_query = xf_query.contiguous().view(1, -1)
        G, G_normalized = self.encode_training_set(xf_support)

        Y_S = Variable(one_hot(y_support.type(torch.LongTensor), self.k_way)).to(self.device)
        f = xf_query

        logprobs = self.get_logprobs(f, G, G_normalized, Y_S)

        return logprobs

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
        
            most_vote = torch.argmax(Tensor(vote))
            y_true.append(clip_artist)
            y_pred.append(most_vote)

            count += 1
            if most_vote == clip_artist:
                correct+=1

        accuracy = correct / float(count)
        f1 = f1_score(y_true, y_pred, average="weighted")

        return accuracy, f1
                
        
    def encode_training_set(self, S, G_encoder = None):
        if G_encoder is None:
            G_encoder = self.G_encoder
        out_G = G_encoder(S.unsqueeze(0))[0]
        out_G = out_G.squeeze(0)
        G = S + out_G[:,:S.size(1)] + out_G[:,S.size(1):]
        G_norm = torch.norm(G, p=2, dim=1).unsqueeze(1).expand_as(G)
        G_normalized = G.div(G_norm + 0.00001)
        return G, G_normalized

    def get_logprobs(self, f, G, G_normalized, Y_S, FCE = None):
        if FCE is None:
            FCE = self.FCE
        F = FCE(f, G)
        F_norm = torch.norm(F, p=2, dim=1).unsqueeze(1).expand_as(F)
        F_normalized = F.div(F_norm+0.00001)
        scores = self.relu(F_normalized.mm(G_normalized.transpose(0,1))) * 100
        softmax = self.softmax(scores)
        logprobs = (softmax.mm(Y_S) + 1e-6).log()
        return logprobs

class FullyContextualEmbedding(nn.Module):
    def __init__(self, feat_dim):
        super(FullyContextualEmbedding, self).__init__()
        self.lstmcell = nn.LSTMCell(feat_dim*2, feat_dim)
        self.softmax = nn.Softmax()
        self.c_0 = Variable(torch.zeros(1,feat_dim))
        self.feat_dim = feat_dim
        #self.K = K

    def forward(self, f, G):
        h = f
        c = self.c_0.expand_as(f)
        G_T = G.transpose(0,1)
        K = G.size(0) #Tuna to be comfirmed
        for k in range(K):
            logit_a = h.mm(G_T)
            a = self.softmax(logit_a)
            r = a.mm(G)
            x = torch.cat((f, r),1)

            h, c = self.lstmcell(x, (h, c))
            h = h + f

        return h

    def to(self, device):
        super(FullyContextualEmbedding, self).to(device)
        self.c_0 = self.c_0.to(device)
        return self
