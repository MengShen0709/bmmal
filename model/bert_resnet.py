import torch
import torch.nn as nn
import torchvision
from transformers import BertModel, BertConfig


def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.01)
    for _, m in module.named_modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


class Classifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_size, output_size, bias=True)
        self.fc.apply(init_weights)

    def forward(self, x):
        logits = self.fc(x)
        return logits, x


class ClassifierSum(nn.Module):
    def __init__(self, m1_feature_size, m2_feature_size, output_size):
        super(ClassifierSum, self).__init__()
        self.fc_m1 = nn.Linear(m1_feature_size, output_size, bias=True)
        self.fc_m1.apply(init_weights)
        self.fc_m2 = nn.Linear(m2_feature_size, output_size, bias=True)
        self.fc_m2.apply(init_weights)

    def forward(self, x1, x2):
        logits_m1 = self.fc_m1(x1)
        logits_m2 = self.fc_m2(x2)
        return logits_m1 + logits_m2, torch.cat([x1, x2], dim=-1)


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101',
                                weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V2)
        for p in resnet.parameters():
            p.requires_grad = True
        self.resnet_feature_size = resnet.fc.in_features
        self.encoder = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.Flatten(start_dim=1),
        )

    def forward(self, image):
        return self.encoder(image)


class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        try:
            self.encoder = BertModel.from_pretrained("bert-base-uncased")
        except ConnectionError:
            self.encoder = BertModel(BertConfig())
            self.encoder.load_state_dict(torch.load("~/.cache/bert_base_uncased.pt"))

    def forward(self, text_input):
        return self.encoder(**text_input).last_hidden_state[:, 0, :]


class BertResnet(nn.Module):
    def __init__(self, m1_feature_size, m2_feature_size, class_num, fuse="concat"):
        super(BertResnet, self).__init__()
        self.fuse = fuse
        self.m1_model = TextEncoder()
        self.m2_model = ImageEncoder()

        if self.fuse == "concat":
            self.mm_classifier = Classifier(m1_feature_size + m2_feature_size, class_num)
        elif self.fuse == "sum":
            self.mm_classifier = ClassifierSum(m1_feature_size, m2_feature_size, class_num)
        else:
            raise NotImplemented(self.fuse)

        self.m1_classifier = Classifier(m1_feature_size, class_num)
        self.m2_classifier = Classifier(m2_feature_size, class_num)

    def feature_extraction(self, m1_input, m2_input):
        m1_feature = self.m1_model(m1_input)
        m2_feature = self.m2_model(m2_input)
        return m1_feature, m2_feature

    def forward(self, m1_feature, m2_feature, return_latent=False):
        text_feature_size = m1_feature.size(1)
        image_feature_size = m2_feature.size(1)

        text_logits, text_feature = self.m1_classifier(m1_feature)
        image_logits, image_feature = self.m2_classifier(m2_feature)
        if self.fuse == "concat":
            concat_mm_features = torch.cat([m1_feature, m2_feature], dim=-1)
            mm_logits, mm_feature = self.mm_classifier(concat_mm_features)
            # text_logits  = m1_feature @ self.mm_classifier.fc.weight[:, :text_feature_size].T \
            #                + self.mm_classifier.fc.bias / 2
            # image_logits = m2_feature @ self.mm_classifier.fc.weight[:, text_feature_size:].T \
            #                + self.mm_classifier.fc.bias / 2
        elif self.fuse == "sum":
            mm_logits = (text_logits + image_logits) / 2
            mm_feature = torch.cat([m1_feature, m2_feature], dim=-1)
        elif self.fuse == "attention":
            mm_logits, mm_feature = self.mm_classifier(text_feature, image_feature)
        else:
            raise NotImplemented(self.fuse)

        if return_latent:
            return text_logits, image_logits, mm_logits, text_feature, image_feature, mm_feature
        else:
            return text_logits, image_logits, mm_logits
