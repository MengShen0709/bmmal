import torch
import torch.nn as nn
from .resnet import generate_resnet
from .torchvision_video import VideoResNetLayer4, r2plus1d_18

def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.01)
    for _, m in module.named_modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


class NLGate(nn.Module):
    def __init__(self, in_channels, inter_channels=None, bn_layer=True):
        super(NLGate, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = in_channels // 2

        self.theta = nn.Conv3d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels=self.in_channels * 2, out_channels=self.inter_channels, kernel_size=1)
        self.g = nn.Conv2d(in_channels=self.in_channels * 2, out_channels=self.inter_channels, kernel_size=1)
        self.h = nn.Conv3d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

    def forward(self, v, a):
        """
        args
            v: (B, C, T, H, W)
            a: (B, C, T, F)
        """

        batch_size = v.size(0)
        channel_size = v.size(1)
        a_t = a.size(2)
        v_t = v.size(2)
        v_h = v.size(3)
        v_w = v.size(4)
        a_f = a.size(3)

        # pool and tile v to (B, C, T, F)
        z = torch.nn.functional.adaptive_avg_pool2d(v, 1)
        z = torch.flatten(z, start_dim=3)
        z = torch.tile(z, dims=(1, 1, 1, a_f))
        z = torch.repeat_interleave(z, repeats=a_t // v_t, dim=2)
        z = torch.cat([z, z[:, :, -1, :].unsqueeze(2).tile((1, 1, a_t % v_t, 1))], dim=2)

        # create concat audio_video feature
        av = torch.cat([a, z], dim=1)  # size (B, 2C, T, F)

        theta_v = self.theta(v).view(batch_size, channel_size // 2, -1).permute(0, 2, 1)  # size (B, THW, C/2)
        phi_av = self.phi(av).view(batch_size, channel_size // 2, -1)  # size (B, C/2, TF)

        f = torch.bmm(theta_v, phi_av)  # size(B, THW, TF)
        f = torch.softmax(f, dim=-1)  # size(B, THW, TF)

        g_av = self.g(av).view(batch_size, channel_size // 2, -1).permute(0, 2, 1)  # size (B, TF, C/2)

        g_av = torch.bmm(f, g_av).permute(0, 2, 1).view(batch_size, channel_size // 2, v_t, v_h, v_w)  # size(B, C/2, T, H, W)
        g_av = self.h(g_av)  # size(B, C, T, H, W)

        return v + g_av



class Classifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_size, output_size, bias=True)
        self.fc.apply(init_weights)

    def forward(self, x):
        logits = self.fc(x)
        return logits, x


class AVModelNLGate(nn.Module):
    def __init__(self, m1_feature_size, m2_feature_size, class_num, fuse="nlgate", pretrain=True):
        super(AVModelNLGate, self).__init__()
        self.m1_model = r2plus1d_18(pretrain=pretrain)
        self.m2_model = generate_resnet(model_depth=18, input_channel=1)
        # self.video_model = generate_video_resnet2p1d(model_depth=18)

        self.nl_gate = NLGate(in_channels=256)
        self.layer4 = VideoResNetLayer4()
        self.mm_classifier = Classifier(m1_feature_size, class_num)
        self.m1_classifier = Classifier(m1_feature_size, class_num)
        self.m2_classifier = Classifier(m2_feature_size, class_num)

    def feature_extraction(self, m1_input, m2_input):
        video_feature, video_z = self.m1_model(m1_input, nl_gate=True)
        audio_feature, audio_z = self.m2_model(m2_input, nl_gate=True)

        return [video_feature, video_z], [audio_feature, audio_z]

    def forward(self, m1_feature, m2_feature, return_latent=False):
        video_feature, video_z = m1_feature
        audio_feature, audio_z = m2_feature
        fused_z = self.nl_gate(video_z, audio_z)
        mm_feature = self.layer4(fused_z)

        mm_logits, z_mm = self.mm_classifier(mm_feature)
        video_logits, _ = self.m1_classifier(video_feature)
        audio_logits, _ = self.m2_classifier(audio_feature)
        if return_latent:
            z_m1 = self.nl_gate(video_z, torch.zeros_like(audio_z))
            z_m1 = self.layer4(z_m1)
            z_m2 = self.nl_gate(torch.zeros_like(video_z), audio_z)
            z_m2 = self.layer4(z_m2)
            return video_logits, audio_logits, mm_logits, z_m1, z_m2, z_mm
        else:
            return video_logits, audio_logits, mm_logits
