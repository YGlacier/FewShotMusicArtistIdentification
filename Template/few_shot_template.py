import torch
from torch.functional import Tensor
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from abc import abstractmethod

class FewShotTemplate:
    def __init__(self, device, k_way, k_shot):
        self.device = device
        self.k_way = k_way
        self.k_shot = k_shot
        self.is_train_log_muted = False

    # output loss
    @abstractmethod
    def forward_loss(self, x, y):
        pass

    # output scores
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def set_train(self):
        pass

    @abstractmethod
    def set_test(self):
        pass

    def train_loop(self, epoch, data_loader, optimizer):
        avg_loss = 0
        print_freq = 10
        for i, data in enumerate(data_loader):
            input = data[2].unsqueeze(1).to(self.device)
            label = data[0].to(self.device)

            optimizer.zero_grad()
            loss = self.forward_loss(input, label)
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

            if not self.is_train_log_muted:
                if i % print_freq == 0:
                    print("Epoch {:d} | Batch {:d}/{:d} | Loss {:f}".format(epoch, i, len(data_loader), avg_loss/float(i+1)))
    
    # return (accuracy, f1_score)
    def test_loop(self, test_clip_data):
        correct = 0
        count = 0
        slice_length = 313
        slice_start_list=[0, 156, 313, 469, 598]
        y_true = []
        y_pred = []

        for clip in test_clip_data:
            clip_spec = clip[2]
            clip_artist = clip[0]

            vote = [0] * self.k_way
            for slice_start in slice_start_list:
                slice_spec = clip_spec[:,slice_start:slice_start+slice_length]
                input=torch.Tensor(slice_spec).unsqueeze(0).unsqueeze(0).to(self.device)
                output = self.forward(input)
                ouput_label = torch.argmax(output)
                vote[ouput_label] += 1

            most_vote = torch.argmax(Tensor(vote))
            y_true.append(clip_artist)
            y_pred.append(most_vote)

            count += 1
            if most_vote == clip_artist:
                correct += 1
        
        accuracy = correct / float(count)
        f1 = f1_score(y_true, y_pred, average="weighted")

        print("Acc: {:f} | F1: {:f}".format(accuracy, f1))
        return accuracy, f1

    def mute_training_log(self):
        self.is_train_log_muted = True

