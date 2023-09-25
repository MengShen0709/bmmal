import torch
import torch.nn as nn
from .resnet2p1d import get_pretrained_resnet2p1d
from .resnet import generate_resnet


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


class VideoEncoder(nn.Module):
    def __init__(self, pretrain):
        super(VideoEncoder, self).__init__()
        if pretrain:
            self.encoder = get_pretrained_resnet2p1d()
        else:
            self.encoder = get_pretrained_resnet2p1d(weights=None)

    def forward(self, video):
        return self.encoder(video)


class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()
        self.encoder = generate_resnet(model_depth=18, input_channel=1)

    def forward(self, audio):
        return self.encoder(audio)


class AVModel(nn.Module):
    def __init__(self, m1_feature_size, m2_feature_size, class_num, fuse="concat", pretrain=True):
        super(AVModel, self).__init__()
        self.m1_model = VideoEncoder(pretrain)
        self.m2_model = AudioEncoder()

        self.fuse = fuse
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
        video_feature_size = m1_feature.size(1)
        audio_feature_size = m2_feature.size(1)

        video_logits, video_feature = self.m1_classifier(m1_feature)
        audio_logits, audio_feature = self.m2_classifier(m2_feature)
        if self.fuse == "concat":
            concat_mm_features = torch.cat([m1_feature, m2_feature], dim=-1)
            mm_logits, mm_feature = self.mm_classifier(concat_mm_features)
            # video_logits = m1_feature @ self.mm_classifier.fc.weight[:, :video_feature_size].T \
            #                + self.mm_classifier.fc.bias / 2
            # audio_logits = m2_feature @ self.mm_classifier.fc.weight[:, video_feature_size:].T \
            #                + self.mm_classifier.fc.bias / 2
        elif self.fuse == "sum":
            mm_logits = (video_logits + audio_logits ) / 2
            mm_feature = torch.cat([m1_feature, m2_feature], dim=-1)
        elif self.fuse == "attention":
            mm_logits, mm_feature = self.mm_classifier(m1_feature, m2_feature)
        else:
            raise NotImplemented(self.fuse)

        if return_latent:
            return video_logits, audio_logits, mm_logits, video_feature, audio_feature, mm_feature
        else:
            return video_logits, audio_logits, mm_logits
