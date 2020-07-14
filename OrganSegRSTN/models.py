import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torchsummary import summary
from torchvision import models
import pandas as pd


class SpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=9):
        super(SpatialAttention, self).__init__()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        pad = (self.kernel_size - 1) // 2  # Padding on one side for stride 1

        self.grp1_conv1k = nn.Conv2d(self.in_channels, self.in_channels // 2, (1, self.kernel_size), padding=(0, pad))
        self.grp1_bn1 = nn.BatchNorm2d(self.in_channels // 2)
        self.grp1_convk1 = nn.Conv2d(self.in_channels // 2, 1, (self.kernel_size, 1), padding=(pad, 0))
        self.grp1_bn2 = nn.BatchNorm2d(1)

        self.grp2_convk1 = nn.Conv2d(self.in_channels, self.in_channels // 2, (self.kernel_size, 1), padding=(pad, 0))
        self.grp2_bn1 = nn.BatchNorm2d(self.in_channels // 2)
        self.grp2_conv1k = nn.Conv2d(self.in_channels // 2, 1, (1, self.kernel_size), padding=(0, pad))
        self.grp2_bn2 = nn.BatchNorm2d(1)

    def forward(self, input_):
        # Generate Group 1 Features
        grp1_feats = self.grp1_conv1k(input_)
        grp1_feats = F.relu(self.grp1_bn1(grp1_feats))
        grp1_feats = self.grp1_convk1(grp1_feats)
        grp1_feats = F.relu(self.grp1_bn2(grp1_feats))

        # Generate Group 2 features
        grp2_feats = self.grp2_convk1(input_)
        grp2_feats = F.relu(self.grp2_bn1(grp2_feats))
        grp2_feats = self.grp2_conv1k(grp2_feats)
        grp2_feats = F.relu(self.grp2_bn2(grp2_feats))

        added_feats = torch.sigmoid(torch.add(grp1_feats, grp2_feats))
        added_feats = added_feats.expand_as(input_).clone()

        return added_feats


class ChannelwiseAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelwiseAttention, self).__init__()

        self.in_channels = in_channels

        self.linear_1 = nn.Linear(self.in_channels, self.in_channels // 4)
        self.linear_2 = nn.Linear(self.in_channels // 4, self.in_channels)

    def forward(self, input_):
        n_b, n_c, h, w = input_.size()

        feats = F.adaptive_avg_pool2d(input_, (1, 1)).view((n_b, n_c))
        feats = F.relu(self.linear_1(feats))
        feats = torch.sigmoid(self.linear_2(feats))

        # Activity regularizer
        ca_act_reg = torch.mean(feats)

        feats = feats.view((n_b, n_c, 1, 1))
        feats = feats.expand_as(input_).clone()

        return feats, ca_act_reg

vgg_conv1_2 = vgg_conv2_2 = vgg_conv3_3 = vgg_conv4_3 = vgg_conv5_3 = None


def conv_1_2_hook(module, input, output):
    global vgg_conv1_2
    vgg_conv1_2 = output
    return None


def conv_2_2_hook(module, input, output):
    global vgg_conv2_2
    vgg_conv2_2 = output
    return None


def conv_3_3_hook(module, input, output):
    global vgg_conv3_3
    vgg_conv3_3 = output
    return None


def conv_4_3_hook(module, input, output):
    global vgg_conv4_3
    vgg_conv4_3 = output
    return None


def conv_5_3_hook(module, input, output):
    global vgg_conv5_3
    vgg_conv5_3 = output
    return None


class CPFE(nn.Module):
    def __init__(self, feature_layer=None, out_channels=32):
        super(CPFE, self).__init__()

        self.dil_rates = [3, 5, 7]

        # Determine number of in_channels from VGG-16 feature layer
        if feature_layer == 'conv5_3':
            self.in_channels = 512
        elif feature_layer == 'conv4_3':
            self.in_channels = 512
        elif feature_layer == 'conv3_3':
            self.in_channels = 256

        # Define layers
        self.conv_1_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.conv_dil_3 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=3,
                                    stride=1, dilation=self.dil_rates[0], padding=self.dil_rates[0], bias=False)
        self.conv_dil_5 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=3,
                                    stride=1, dilation=self.dil_rates[1], padding=self.dil_rates[1], bias=False)
        self.conv_dil_7 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channels, kernel_size=3,
                                    stride=1, dilation=self.dil_rates[2], padding=self.dil_rates[2], bias=False)

        self.bn = nn.BatchNorm2d(out_channels*4)

    def forward(self, input_):
        # Extract features
        conv_1_1_feats = self.conv_1_1(input_)
        conv_dil_3_feats = self.conv_dil_3(input_)
        conv_dil_5_feats = self.conv_dil_5(input_)
        conv_dil_7_feats = self.conv_dil_7(input_)

        # Aggregate features
        concat_feats = torch.cat((conv_1_1_feats, conv_dil_3_feats, conv_dil_5_feats, conv_dil_7_feats), dim=1)
        bn_feats = F.relu(self.bn(concat_feats))

        return bn_feats


class SODModel(nn.Module):
    def __init__(self):
        super(SODModel, self).__init__()

        # Load the [partial] VGG-16 model
        self.vgg16 = models.vgg16(pretrained=True).features

        # Extract and register intermediate features of VGG-16
        self.vgg16[3].register_forward_hook(conv_1_2_hook)
        self.vgg16[8].register_forward_hook(conv_2_2_hook)
        self.vgg16[15].register_forward_hook(conv_3_3_hook)
        self.vgg16[22].register_forward_hook(conv_4_3_hook)
        self.vgg16[29].register_forward_hook(conv_5_3_hook)

        # Initialize layers for high level (hl) feature (conv3_3, conv4_3, conv5_3) processing
        self.cpfe_conv3_3 = CPFE(feature_layer='conv3_3')
        self.cpfe_conv4_3 = CPFE(feature_layer='conv4_3')
        self.cpfe_conv5_3 = CPFE(feature_layer='conv5_3')

        self.cha_att = ChannelwiseAttention(in_channels=384)  # in_channels = 3 x (32 x 4)

        self.hl_conv1 = nn.Conv2d(384, 64, (3, 3), padding=1)
        self.hl_bn1 = nn.BatchNorm2d(64)

        # Initialize layers for low level (ll) feature (conv1_2 and conv2_2) processing
        self.ll_conv_1 = nn.Conv2d(64, 64, (3, 3), padding=1)
        self.ll_bn_1 = nn.BatchNorm2d(64)
        self.ll_conv_2 = nn.Conv2d(128, 64, (3, 3), padding=1)
        self.ll_bn_2 = nn.BatchNorm2d(64)
        self.ll_conv_3 = nn.Conv2d(128, 64, (3, 3), padding=1)
        self.ll_bn_3 = nn.BatchNorm2d(64)

        self.spa_att = SpatialAttention(in_channels=64)

        # Initialize layers for fused features (ff) processing
        self.ff_conv_1 = nn.Conv2d(128, 3, (3, 3), padding=1)

    def forward(self, input_):
        global vgg_conv1_2, vgg_conv2_2, vgg_conv3_3, vgg_conv4_3, vgg_conv5_3

        # Pass input_ through vgg16 to generate intermediate features
        self.vgg16(input_)
        # print(vgg_conv1_2.size())
        # print(vgg_conv2_2.size())
        # print(vgg_conv3_3.size())
        # print(vgg_conv4_3.size())
        # print(vgg_conv5_3.size())

        # Process high level features
        conv3_cpfe_feats = self.cpfe_conv3_3(vgg_conv3_3)
        conv4_cpfe_feats = self.cpfe_conv4_3(vgg_conv4_3)
        conv5_cpfe_feats = self.cpfe_conv5_3(vgg_conv5_3)

        conv4_cpfe_feats = F.interpolate(conv4_cpfe_feats, size=[conv3_cpfe_feats.size(2),conv3_cpfe_feats.size(3)], mode='bilinear', align_corners=True)
        conv5_cpfe_feats = F.interpolate(conv5_cpfe_feats, size=[conv3_cpfe_feats.size(2),conv3_cpfe_feats.size(3)], mode='bilinear', align_corners=True)

        conv_345_feats = torch.cat((conv3_cpfe_feats, conv4_cpfe_feats, conv5_cpfe_feats), dim=1)

        conv_345_ca, ca_act_reg = self.cha_att(conv_345_feats)
        conv_345_feats = torch.mul(conv_345_feats, conv_345_ca)

        conv_345_feats = self.hl_conv1(conv_345_feats)
        conv_345_feats = F.relu(self.hl_bn1(conv_345_feats))
        conv_345_feats = F.interpolate(conv_345_feats, scale_factor=4, mode='bilinear', align_corners=True)

        # Process low level features
        conv1_feats = self.ll_conv_1(vgg_conv1_2)
        conv1_feats = F.relu(self.ll_bn_1(conv1_feats))
        conv2_feats = self.ll_conv_2(vgg_conv2_2)
        conv2_feats = F.relu(self.ll_bn_2(conv2_feats))

        conv2_feats = F.interpolate(conv2_feats, size=[conv1_feats.size(2),conv1_feats.size(3)], mode='bilinear', align_corners=True)
        conv_12_feats = torch.cat((conv1_feats, conv2_feats), dim=1)
        conv_12_feats = self.ll_conv_3(conv_12_feats)
        conv_12_feats = F.relu(self.ll_bn_3(conv_12_feats))

        conv_12_sa = self.spa_att(conv_345_feats)
        conv_12_sa = F.interpolate(conv_12_sa, size=[conv_12_feats.size(2), conv_12_feats.size(3)], mode='bilinear',align_corners=True)
        conv_12_feats = torch.mul(conv_12_feats, conv_12_sa)

        # Fused features
        conv_345_feats = F.interpolate(conv_345_feats, size=[conv_12_feats.size(2), conv_12_feats.size(3)], mode='bilinear',align_corners=True)
        fused_feats = torch.cat((conv_12_feats, conv_345_feats), dim=1)
        fused_feats = torch.sigmoid(self.ff_conv_1(fused_feats))

        return fused_feats, ca_act_reg

class RSTN(nn.Module):
    def __init__(self, crop_margin=40, crop_prob=0.5, \
                 crop_sample_batch=1, n_class=3, TEST=None):
        super(RSTN, self).__init__()
        self.TEST = TEST
        self.margin = crop_margin
        self.prob = crop_prob
        self.batch = crop_sample_batch


        # Coarse-scaled Network
        self.coarse_model = SODModel()
        # Saliency Transformation Module

        self.saliency1 = nn.Conv2d(n_class, n_class, kernel_size=3, stride=1, padding=1)
        self.relu_saliency1 = nn.ReLU(inplace=True)
        self.saliency2 = nn.Conv2d(n_class, n_class, kernel_size=5, stride=1, padding=2)
        # Fine-scaled Network

        self.fine_model = SODModel()
        # self.fine_model = UNet()

        self._initialize_weights()

    def _initialize_weights(self):
        for name, mod in self.named_children():
            if name == 'saliency1':
                nn.init.xavier_normal_(mod.weight.data)
                mod.bias.data.fill_(1)
            elif name == 'saliency2':
                mod.weight.data.zero_()
                mod.bias.data = torch.tensor([1.0, 1.5, 2.0])

    def forward(self, image, label=None, mode=None, score=None, mask=None):
        if self.TEST is None:
            assert label is not None and mode is not None \
                   and score is None and mask is None
            # Coarse-scaled Network
            h = image
            h = self.coarse_model(h)
            h = torch.sigmoid(h)
            coarse_prob = h
            # Saliency Transformation Module
            h = self.relu_saliency1(self.saliency1(h))
            h = self.saliency2(h)
            saliency = h

            if mode == 'S':
                cropped_image, crop_info = self.crop(label, image)
            elif mode == 'I':
                cropped_image, crop_info = self.crop(label, image * saliency)
            elif mode == 'J':
                cropped_image, crop_info = self.crop(coarse_prob, image * saliency, label)
            else:
                raise ValueError("wrong value of mode, should be in ['S', 'I', 'J']")

            # Fine-scaled Network
            k = cropped_image
            k = self.fine_model(k)


            k = self.uncrop(crop_info, k, image)
            k = torch.sigmoid(k)
            fine_prob = k
            return coarse_prob, fine_prob

        elif self.TEST == 'C':  # Coarse testing
            assert label is None and mode is None and \
                   score is None and mask is None
            # Coarse-scaled Network
            h = image
            h = self.coarse_model(h)
            h = torch.sigmoid(h)
            coarse_prob = h
            return coarse_prob

        elif self.TEST == 'O':  # Oracle testing
            assert label is not None and mode is None and \
                   score is None and mask is None
            # Coarse-scaled Network
            h = image
            h = self.coarse_model(h)
            h = torch.sigmoid(h)
            # Saliency Transformation Module
            h = self.relu_saliency1(self.saliency1(h))
            h = self.saliency2(h)
            saliency = h
            cropped_image, crop_info = self.crop(label, image * saliency)
            # Fine-scaled Network
            h = cropped_image
            h = self.fine_model(h)
            h = self.uncrop(crop_info, h, image)
            h = torch.sigmoid(h)
            fine_prob = h
            return fine_prob

        elif self.TEST == 'F':  # Fine testing
            assert label is None and mode is None \
                   and score is not None and mask is not None
            # Saliency Transformation Module
            h = score
            h = self.relu_saliency1(self.saliency1(h))
            h = self.saliency2(h)
            saliency = h
            cropped_image, crop_info = self.crop(mask, image * saliency)
            # Fine-scaled Network
            h = cropped_image
            h = self.fine_model(h)
            h = self.uncrop(crop_info, h, image)
            h = torch.sigmoid(h)
            fine_prob = h
            return fine_prob

        else:
            raise ValueError("wrong value of TEST, should be in [None, 'C', 'F', 'O']")

    def crop(self, prob_map, saliency_data, label=None):
        (N, C, W, H) = prob_map.shape

        binary_mask = (prob_map >= 0.5)  # torch.uint8
        if label is not None and binary_mask.sum().item() == 0:
            binary_mask = (label >= 0.5)

        if self.TEST is not None:
            self.left = self.margin
            self.right = self.margin
            self.top = self.margin
            self.bottom = self.margin
        else:
            self.update_margin()

        if binary_mask.sum().item() == 0:  # avoid this by pre-condition in TEST 'F'
            minA = 0
            maxA = W
            minB = 0
            maxB = H
            self.no_forward = True
        else:
            if N > 1:
                mask = torch.zeros(size=(N, C, W, H))
                for n in range(N):
                    cur_mask = binary_mask[n, :, :, :]
                    arr = torch.nonzero(cur_mask)
                    minA = arr[:, 1].min().item()
                    maxA = arr[:, 1].max().item()
                    minB = arr[:, 2].min().item()
                    maxB = arr[:, 2].max().item()
                    bbox = [int(max(minA - self.left, 0)), int(min(maxA + self.right + 1, W)), \
                            int(max(minB - self.top, 0)), int(min(maxB + self.bottom + 1, H))]
                    mask[n, :, bbox[0]: bbox[1], bbox[2]: bbox[3]] = 1
                saliency_data = saliency_data * mask.cuda()

            arr = torch.nonzero(binary_mask)
            minA = arr[:, 2].min().item()
            maxA = arr[:, 2].max().item()
            minB = arr[:, 3].min().item()
            maxB = arr[:, 3].max().item()
            self.no_forward = False

        bbox = [int(max(minA - self.left, 0)), int(min(maxA + self.right + 1, W)), \
                int(max(minB - self.top, 0)), int(min(maxB + self.bottom + 1, H))]
        cropped_image = saliency_data[:, :, bbox[0]: bbox[1], \
                        bbox[2]: bbox[3]]

        if self.no_forward == True and self.TEST == 'F':
            cropped_image = torch.zeros_like(cropped_image).cuda()

        crop_info = np.zeros((1, 4), dtype=np.int16)
        crop_info[0] = bbox
        crop_info = torch.from_numpy(crop_info).cuda()

        return cropped_image, crop_info

    def update_margin(self):
        MAX_INT = 256
        if random.randint(0, MAX_INT - 1) >= MAX_INT * self.prob:
            self.left = self.margin
            self.right = self.margin
            self.top = self.margin
            self.bottom = self.margin
        else:
            a = np.zeros(self.batch * 4, dtype=np.uint8)
            for i in range(self.batch * 4):
                a[i] = random.randint(0, self.margin * 2)
            self.left = int(a[0: self.batch].sum() / self.batch)
            self.right = int(a[self.batch: self.batch * 2].sum() / self.batch)
            self.top = int(a[self.batch * 2: self.batch * 3].sum() / self.batch)
            self.bottom = int(a[self.batch * 3: self.batch * 4].sum() / self.batch)

    def uncrop(self, crop_info, cropped_image, image):
        uncropped_image = torch.ones_like(image).cuda()
        uncropped_image *= (-9999999)
        bbox = crop_info[0]
        uncropped_image[:, :, bbox[0].item(): bbox[1].item(), bbox[2].item(): bbox[3].item()] = cropped_image
        return uncropped_image


def get_parameters(model, coarse=True, bias=False, parallel=False):
    print('coarse, bias', coarse, bias)
    if parallel:
        for name, mod in model.named_children():
            print('parallel', name)
            model = mod
            break
    for name, mod in model.named_children():
        if name == 'coarse_model' and coarse \
                or name in ['saliency1', 'saliency2', 'fine_model'] and not coarse:
            print(name)
            for n, m in mod.named_modules():
                if isinstance(m, nn.Conv2d):
                    print(n, m)
                    if bias and m.bias is not None:
                        yield m.bias
                    elif not bias:
                        yield m.weight
                elif isinstance(m, nn.ConvTranspose2d):
                    # weight is frozen because it is just a bilinear upsampling
                    if bias:
                        assert m.bias is None


class DSC_loss(nn.Module):
    def __init__(self):
        super(DSC_loss, self).__init__()
        self.epsilon = 0.000001
        return

    def forward(self, pred, target):  # soft mode. per item.
        batch_num = pred.shape[0]
        pred = pred.contiguous().view(batch_num, -1)
        target = target.contiguous().view(batch_num, -1)
        DSC = (2 * (pred * target).sum(1) + self.epsilon) / ((pred + target).sum(1) + self.epsilon)
        return 1 - DSC.sum() / float(batch_num)


if __name__ == '__main__':
    RSTN_model = RSTN(crop_margin=20, crop_prob=0.5, crop_sample_batch=1)
    RSTN_dict = RSTN_model.state_dict()
    model_parameters = filter(lambda p: p.requires_grad, RSTN_model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('model parameters:', params)
    for name, param in RSTN_model.named_parameters():
        if param.requires_grad:
            print(name)
    RSTN_dict = RSTN_model.state_dict()
    pretrained_dict = torch.load('INTERRUPT.pth')  # FCN8s(n_class=21)
    # 1. filter out unnecessary keys
    pretrained_dict_coarse = {'coarse_model.' + k: v
                              for k, v in pretrained_dict.items()
                              if 'coarse_model.' + k in RSTN_dict and 'score' not in k}
    pretrained_dict_fine = {'fine_model.' + k: v
                            for k, v in pretrained_dict.items()
                            if 'fine_model.' + k in RSTN_dict and 'score' not in k}
    # 2. overwrite entries in the existing state dict
    RSTN_dict.update(pretrained_dict_coarse)
    RSTN_dict.update(pretrained_dict_fine)
    # 3. load the new state dict
    RSTN_model.load_state_dict(RSTN_dict)
    optimizer = torch.optim.SGD(
        [
            {'params': get_parameters(RSTN_model, coarse=True, bias=False, parallel=False)},
            {'params': get_parameters(RSTN_model, coarse=True, bias=True, parallel=False),
             'lr': (1e-5) * 2, 'weight_decay': 0},
            {'params': get_parameters(RSTN_model, coarse=False, bias=False, parallel=False),
             'lr': (1e-5) * 10},
            {'params': get_parameters(RSTN_model, coarse=False, bias=True, parallel=False),
             'lr': (1e-5) * 20, 'weight_decay': 0}
        ],
        lr=1e-5,
        momentum=0.99,
        weight_decay=0.0005)
    print(optimizer.param_groups)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = RSTN().to(device)
    summary(unet, (3, 64, 64))
