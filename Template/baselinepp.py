import torch
from .few_shot_template import FewShotTemplate
from .model import ArtistIdentificationFeatureModel, ArtistIdentificationDistClassifierModel

class BaselinePP(FewShotTemplate):
    def __init__(self, device, k_way, k_shot):
        super(BaselinePP, self).__init__(device, k_way, k_shot)
        self.feature_model = ArtistIdentificationFeatureModel().to(self.device)
        self.feature_model.eval()
        self.classifier_model = ArtistIdentificationDistClassifierModel(self.k_way).to(self.device)

        self.loss_function = torch.nn.CrossEntropyLoss()

    def set_train(self):
        self.classifier_model.train()
    
    def set_test(self):
        self.classifier_model.eval()

    def forward(self, x):
        with torch.no_grad():
            features = self.feature_model(x)
        scores = self.classifier_model(features)
        return scores
    
    def forward_loss(self, x, y):
        scores = self.forward(x)
        loss = self.loss_function(scores, y)
        return loss

    def load_feature_weight(self, path):
        self.feature_model.load_state_dict(torch.load(path))

    def load_classifier_weight(self, path):
        self.classifier_model.load_state_dict(torch.load(path))

    def save_classifier_weight(self, path):
        torch.save(self.classifier_model.state_dict(), path)

    def reset_classifier_weight(self):
        self.classifier_model.reset_parameters()