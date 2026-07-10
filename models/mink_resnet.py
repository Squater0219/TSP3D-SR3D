# Copyright (c) OpenMMLab. All rights reserved.
# Follow https://github.com/NVIDIA/MinkowskiEngine/blob/master/examples/resnet.py # noqa
# and mmcv.cnn.ResNet
try:
    import MinkowskiEngine as ME
    from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
except ImportError:
    import warnings
    warnings.warn(
        'Please follow `getting_started.md` to install MinkowskiEngine.`')
    # blocks are used in the static part of MinkResNet
    BasicBlock, Bottleneck = None, None

import torch.nn as nn


class TSPBackbone(nn.Module):
    r"""Minkowski ResNet backbone. See `4D Spatio-Temporal ConvNets
    <https://arxiv.org/abs/1904.08755>`_ for more details.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (ont): Number of input channels, 3 for RGB.
        num_stages (int, optional): Resnet stages. Default: 4.
        pool (bool, optional): Add max pooling after first conv if True.
            Default: True.
    """
    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth=34,
                 in_channels=3,
                 max_channels=128,
                 num_stages=4,
                 pool=True,
                 norm='batch'):
        super().__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        assert 4 >= num_stages >= 1
        block, stage_blocks = self.arch_settings[depth]
        stage_blocks = stage_blocks[:num_stages]
        self.max_channels = max_channels
        self.num_stages = num_stages
        self.pool = pool

        self.inplanes = 64
        self.conv1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=3, stride=2, dimension=3)
        norm1 = ME.MinkowskiInstanceNorm if norm == 'instance' \
            else ME.MinkowskiBatchNorm
        self.norm1 = norm1(self.inplanes)
        self.relu = ME.MinkowskiReLU(inplace=True)
        if self.pool:
            self.maxpool = ME.MinkowskiMaxPooling(
                kernel_size=2, stride=2, dimension=3)

        for i, _ in enumerate(stage_blocks):
            n_channels = 64 * 2**i
            if self.max_channels is not None:
                n_channels = min(n_channels, self.max_channels)
            setattr(
                self, f'layer{i + 1}',
                self._make_layer(block, n_channels, stage_blocks[i], stride=2))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(
                    m.kernel, mode='fan_out', nonlinearity='relu')

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _make_layer(self, block, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                ME.MinkowskiConvolution(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    dimension=3),
                ME.MinkowskiBatchNorm(planes * block.expansion))
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                dimension=3))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, dimension=3))
        return nn.Sequential(*layers)

    def forward(self, x):
        #Step 5-1: Stem — conv1(k=3,s=2) + BN + ReLU로 초기 특징 추출
        # SparseTensor(N_voxels, 6) → SparseTensor(N_voxels, 64), tensor_stride=2, 격자=0.02m
        # 6채널(xyz+RGB)을 64채널로 확장. stride=2로 voxel 수 감소.
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        if self.pool:
            #Step 5-2: MaxPool(k=2,s=2) — 공간 해상도를 절반으로 줄여 수용장 확대
            # SparseTensor(N_voxels, 64) → SparseTensor(N_voxels↓, 64), tensor_stride=4, 격자=0.04m
            # Sparse MaxPooling: 각 2×2×2 블록 내 최댓값만 보존. Stem 후 누적 stride=4.
            x = self.maxpool(x)

        outs = []
        for i in range(self.num_stages):
            #Step 5-3~5-6: ResNet Stage {i+1} — BasicBlock×n으로 깊이별 다중 스케일 특징 추출
            # 각 스테이지 시작의 stride=2 다운샘플 → 이후 블록은 stride=1 유지
            # i=0: BasicBlock×3, ch=64,  tensor_stride=8,  격자=0.08m → outs[0] (TSPHead 미사용)
            # i=1: BasicBlock×4, ch=128, tensor_stride=16, 격자=0.16m → outs[1] (inputs[0], Completion 소스)
            # i=2: BasicBlock×6, ch=128, tensor_stride=32, 격자=0.32m → outs[2] (inputs[1], 레벨1)
            # i=3: BasicBlock×3, ch=128, tensor_stride=64, 격자=0.64m → outs[3] (inputs[2], 레벨2 시작)
            x = getattr(self, f'layer{i + 1}')(x)
            outs.append(x)
        return outs


