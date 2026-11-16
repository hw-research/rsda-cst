"""
DeepLabV3+ for remote sensing segmentation.
- Supports return_feat for prototype alignment and student fusion.
- Modified: return_feat returns (losses, outputs, aspp_feat, low_level_feat)
- Added forward_decoder() for student-side feature fusion.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.torch_compat import torch_load_compat
from torchvision import models

BatchNorm = nn.SyncBatchNorm


# ------------------- ASPP Module -------------------
class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        self.branches = nn.ModuleList()

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            BatchNorm(out_channels),
            nn.ReLU(inplace=True)
        ))

        for rate in atrous_rates:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                BatchNorm(out_channels),
                nn.ReLU(inplace=True)
            ))

        self.branches.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=True),
            nn.ReLU(inplace=True)
        ))

        fusion_in_channels = out_channels * (len(atrous_rates) + 2)
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in_channels, out_channels, 1, bias=False),
            BatchNorm(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for branch in self.branches:
            out = branch(x)
            if out.size() != x.size():
                out = F.interpolate(out, size=x.size()[2:], mode='bilinear', align_corners=False)
            res.append(out)
        out = torch.cat(res, dim=1)
        out = self.fusion(out)
        return out


# ------------------- Decoder Module -------------------
class Decoder(nn.Module):
    def __init__(self, low_level_channels, num_low_level, num_classes):
        super(Decoder, self).__init__()
        self.conv_low = nn.Sequential(
            nn.Conv2d(low_level_channels, num_low_level, 1, bias=False),
            BatchNorm(num_low_level),
            nn.ReLU(inplace=True)
        )
        self.last_conv = nn.Sequential(
            nn.Conv2d(256 + num_low_level, 256, 3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv_low(low_level_feat)
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, low_level_feat], dim=1)
        x = self.last_conv(x)
        return x


# ------------------- DeepLabV3Plus (Modified) -------------------
class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes, criterion=None, pretrained=None, backbone='resnet50',
                 freeze_bn=False, aspp_rates=(6, 12, 18), output_stride=8, **kwargs):
        super(DeepLabV3Plus, self).__init__()

        if type(output_stride) is not int or output_stride not in (8, 16):
            raise ValueError("Output stride must be 8 or 16")
        if (not isinstance(aspp_rates, (list, tuple)) or not aspp_rates
                or any(type(rate) is not int or rate <= 0 for rate in aspp_rates)):
            raise ValueError("ASPP rates must be a non-empty list of positive integers")

        self.backbone_name = backbone
        self.num_classes = num_classes
        self.output_stride = output_stride
        stride_dilation = [False, output_stride == 8, True]

        if backbone == 'resnet101':
            try:
                self.backbone = models.resnet101(weights=None, replace_stride_with_dilation=stride_dilation)
            except TypeError:
                self.backbone = models.resnet101(pretrained=False, replace_stride_with_dilation=stride_dilation)
            feat_dim = 2048
            low_level_dim = 256
        elif backbone == 'resnet50':
            try:
                # 新版本 TorchVision (>=0.13) 使用 weights 参数
                self.backbone = models.resnet50(weights=None, replace_stride_with_dilation=stride_dilation)
            except TypeError:
                # 旧版本使用 pretrained 参数
                self.backbone = models.resnet50(pretrained=False, replace_stride_with_dilation=stride_dilation)
            feat_dim = 2048
            low_level_dim = 256
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.backbone.fc = nn.Identity()

        if pretrained is not None and isinstance(pretrained, str) and os.path.isfile(pretrained):
            print(f"Loading backbone weights from: {pretrained}")
            state_dict = torch_load_compat(
                pretrained, map_location='cpu', weights_only=False
            )
            if all(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            self.backbone.load_state_dict(state_dict, strict=False)
            print("Backbone weights loaded successfully.")
        else:
            print(f"Warning: Pretrained weights not found at '{pretrained}'. Backbone randomly initialized.")

        self.aspp = ASPP(feat_dim, aspp_rates)
        self.decoder = Decoder(low_level_dim, 48, num_classes)

        self._init_new_layers()
        self._from_scratch(self.aspp)
        self._from_scratch(self.decoder)

        if freeze_bn:
            self._freeze_bn(self)

        if criterion is not None:
            if hasattr(criterion, 'weight') and criterion.weight is not None:
                if len(criterion.weight) != num_classes:
                    print(f"Warning: criterion weight length {len(criterion.weight)} != num_classes {num_classes}. Using unweighted loss.")
                    ignore_idx = getattr(criterion, 'ignore_index', 255)
                    self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_idx)
                else:
                    self.criterion = criterion
            else:
                self.criterion = criterion
        else:
            self.criterion = None

        print(f"DeepLabV3+ initialized with backbone: {backbone}, feat_dim: {feat_dim}")

    def _init_new_layers(self):
        for m in self.aspp.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, BatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, BatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _from_scratch(self, module):
        if hasattr(module, 'from_scratch'):
            module.from_scratch = True
        for child in module.children():
            self._from_scratch(child)

    def _freeze_bn(self, module):
        for m in module.modules():
            if isinstance(m, BatchNorm):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def parameter_groups(self, base_lr, wd):
        old_lr_mult, new_lr_mult = self.lr_mult()
        old_lr_mult_bias, new_lr_mult_bias = self.lr_mult_bias()

        backbone_params = []
        backbone_bias = []
        new_params = []
        new_bias = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            is_new = any(part in name for part in ['aspp', 'decoder'])
            if is_new:
                if len(param.shape) == 1:
                    new_bias.append(param)
                else:
                    new_params.append(param)
            else:
                if len(param.shape) == 1:
                    backbone_bias.append(param)
                else:
                    backbone_params.append(param)

        groups = [
            {'params': backbone_params, 'lr': base_lr * old_lr_mult, 'weight_decay': wd},
            {'params': backbone_bias, 'lr': base_lr * old_lr_mult_bias, 'weight_decay': 0},
            {'params': new_params, 'lr': base_lr * new_lr_mult, 'weight_decay': wd},
            {'params': new_bias, 'lr': base_lr * new_lr_mult_bias, 'weight_decay': 0},
        ]
        return groups

    def lr_mult(self):
        return 1., 10.

    def lr_mult_bias(self):
        return 2., 20.

    def forward(self, x, y=None, return_feat=False):
        input_shape = x.shape[-2:]

        # Backbone forward
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        low_level_feat = x
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # ASPP (deep feature)
        aspp_feat = self.aspp(x)

        # Decoder
        logits = self.decoder(aspp_feat, low_level_feat)
        logits_up = F.interpolate(logits, size=input_shape, mode='bilinear', align_corners=True)

        losses = {}
        net_outs = {'logits_up': logits_up, 'logits': logits}

        if y is not None and self.criterion is not None:
            loss = self.criterion(logits_up, y)
            losses['loss_ce'] = loss.mean().view(1)

        # ========== 支持返回特征（用于原型对齐或学生端融合） ==========
        if return_feat:
            # 返回：losses, outputs, aspp_feat, low_level_feat
            return losses, net_outs, aspp_feat, low_level_feat
        elif y is None:
            return logits, logits_up
        else:
            return losses, net_outs

    def forward_decoder(self, aspp_feat, low_level_feat, input_shape):
        """
        专门用于学生端融合后重新生成 logits。
        参数:
            aspp_feat: 融合后的深层特征 (B, 256, H/16, W/16)
            low_level_feat: 原始浅层特征 (B, 256, H/4, W/4) - 来自 backbone layer1
            input_shape: (H, W) 原始图像尺寸
        返回:
            logits_up: 上采样后的分割图 (B, num_classes, H, W)
        """
        logits = self.decoder(aspp_feat, low_level_feat)
        logits_up = F.interpolate(logits, size=input_shape, mode='bilinear', align_corners=True)
        return logits_up

    def get_backbone_name(self):
        return self.backbone_name
