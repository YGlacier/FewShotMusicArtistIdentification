import torch
import torch.nn as nn

class ArtistIdentificationModel(nn.Module):
    def __init__(self):
        super(ArtistIdentificationModel, self).__init__()

        self.model_cnn = nn.Sequential()
        self.model_rnn = nn.Sequential()
        self.model_dense = nn.Sequential()

        # Conv2d - Activation - BatchNorm - MaxPool2d - Dropout
        self.model_cnn.add_module("Conv2d_1", nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3,3), padding_mode='replicate'))
        self.model_cnn.add_module("ELU_1", nn.ELU())
        self.model_cnn.add_module("BatchNorm_1", nn.BatchNorm2d(64))
        self.model_cnn.add_module("MaxPool2d_1", nn.MaxPool2d((2,2)))
        self.model_cnn.add_module("Dropout_1", nn.Dropout(0.1))
        self.model_cnn.add_module("Conv2d_2", nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding_mode='replicate'))
        self.model_cnn.add_module("ELU_2", nn.ELU())
        self.model_cnn.add_module("BatchNorm_2", nn.BatchNorm2d(128))
        self.model_cnn.add_module("MaxPool2d_2", nn.MaxPool2d((2,2)))
        self.model_cnn.add_module("Dropout_2", nn.Dropout(0.1))
        self.model_cnn.add_module("Conv2d_3", nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding_mode='replicate'))
        self.model_cnn.add_module("ELU_3", nn.ELU())
        self.model_cnn.add_module("BatchNorm_3", nn.BatchNorm2d(128))
        self.model_cnn.add_module("MaxPool2d_3", nn.MaxPool2d((2,2)))
        self.model_cnn.add_module("Dropout_3", nn.Dropout(0.1))
        self.model_cnn.add_module("Conv2d_4", nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding_mode='replicate'))
        self.model_cnn.add_module("ELU_4", nn.ELU())
        self.model_cnn.add_module("BatchNorm_4", nn.BatchNorm2d(128))
        self.model_cnn.add_module("MaxPool2d_4", nn.MaxPool2d((2,2)))
        self.model_cnn.add_module("Dropout_4", nn.Dropout(0.1))

        # GRU - GRU - Dropout
        self.model_rnn.add_module("GRU", nn.GRU(input_size= 128*6, hidden_size=32, num_layers=2, batch_first=True, dropout=0.3))

        # Dense - Softmax
        self.model_dense.add_module("Dense", nn.Linear(17 * 32, 4))
        self.model_dense.add_module("SoftMax", nn.Softmax(dim=1))

    def forward(self, input):
        cnn_output = self.model_cnn(input)
        cnn_output = cnn_output.view(cnn_output.shape[0], -1, cnn_output.shape[3])
        cnn_output = torch.swapaxes(cnn_output, 1, 2)
        rnn_output = self.model_rnn(cnn_output)[0]
        rnn_output = rnn_output.reshape(rnn_output.shape[0], -1)
        return self.model_dense(rnn_output)