import numpy as np

import MinkowskiEngine as ME

import torch,time
import torch.nn.functional as F
from mmcv.ops import nms3d, nms3d_normal
from torch import nn

from mmdet3d.structures.bbox_3d import rotation_3d_in_axis
from .axis_aligned_iou_loss import (
    AxisAlignedIoULoss2, axis_aligned_bbox_overlaps_3d, axis_aligned_diou_loss)
from mmdet.models.losses import FocalLoss
from .trans_modules import (BiEncoder, BiEncoderLayer, PositionEmbeddingLearned)

import pdb
import logging


def soft_rank(scores, tau=0.1, eps=1e-6):
    """
    내림차순 soft rank. SR3D 보충자료 Eq.9-10.
    scores: 1D tensor (점수가 클수록 '좋음', rank가 높음)
    반환 r: 1D tensor, r in (0,1], 점수가 클수록 r이 1에 가까움.
      R_i = (1/(N-1)) * sum_{j != i} sigmoid((s_j - s_i)/tau)   # 나보다 큰 점수의 비율
      r_i = exp(-R_i)
    tau -> 0 이면 hard rank에 수렴. 작은 점수일수록 R_i가 커져 r_i가 작아진다.
    """
    if scores.numel() == 0:
        return scores
    s = scores.view(-1)
    N = s.numel()
    if N == 1:
        return torch.ones_like(s)
    diff = s.unsqueeze(0) - s.unsqueeze(1)          # diff[i, j] = s_j - s_i
    mask = ~torch.eye(N, dtype=torch.bool, device=s.device)
    # i(행) 고정, j(열)에 대해 합산 -> R_i = sum_j sigmoid((s_j - s_i)/tau) / (N-1)
    R = (torch.sigmoid(diff / tau) * mask).sum(dim=1) / (N - 1)
    r = torch.exp(-R)
    return r

class MinkowskiFeatureFusionBlock(nn.Module):
    """
    Block to fuse backbone features with text features in Minkowski space.
    """
    def __init__(self, backbone_channels, text_channels, output_channels, dimension=3):
        super(MinkowskiFeatureFusionBlock, self).__init__()
        self.conv = ME.MinkowskiConvolution(
            backbone_channels + text_channels,
            output_channels,
            kernel_size=1,
            stride=1,
            dimension=dimension
        )
        self.norm = ME.MinkowskiBatchNorm(output_channels)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, backbone_feats, text_feats):
        # Extract batch indices from the coordinates of backbone features
        batch_indices = backbone_feats.C[:, 0].long()  # Last column is batch index
        
        # Repeat text features for each point in the corresponding batch
        repeated_text_feats = text_feats[batch_indices]  # Use indexing to repeat text features
        
        # Combine the backbone and text features
        combined_features = torch.cat([backbone_feats.F, repeated_text_feats], dim=1)
        combined_feats = ME.SparseTensor(
            features=combined_features,
            coordinate_map_key=backbone_feats.coordinate_map_key,
            coordinate_manager=backbone_feats.coordinate_manager
        )
        
        # Convolution and normalization
        x = self.conv(combined_feats)
        x = self.norm(x)
        return self.relu(x)
    
def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to giving probablity."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init

class TSPHead(nn.Module):
    def __init__(self,
                 n_classes=1,
                 in_channels=(128, 128, 128),
                 out_channels=128,
                 n_reg_outs=6,
                 voxel_size=.01,
                 pts_prune_threshold=(1200,4000),
                 top_pts_threshold=32,
                 volume_threshold=27,
                 r=(13,13),
                 assign_type='volume',
                 prune_threshold=(0.3,0.7),
                 com_threshold = 0.15,
                 train_cfg=None,
                 test_cfg=dict(nms_pre=1, iou_thr=.5, score_thr=.01),
                 keep_loss_weight = 1.0,
                 bbox_loss_weight = 1.0,
                 use_spota=False, use_ras=False,
                 spota_k=6, spota_mu=1.0, spota_alpha=0.0,
                 ras_beta=1.0, ras_tau=0.1):
        super(TSPHead, self).__init__()
        self.voxel_size = voxel_size
        self.pts_prune_threshold = pts_prune_threshold
        self.assign_type = assign_type
        self.volume_threshold = volume_threshold
        self.r = r
        self.prune_threshold = prune_threshold
        self.keep_loss_weight = keep_loss_weight
        self.bbox_loss_weight = bbox_loss_weight
        # SR3D SPOTA/RAS ablation 플래그 — 전부 기본 off. off일 때 기존 baseline과 동일 동작.
        self.use_spota = use_spota
        self.use_ras = use_ras
        self.spota_k = spota_k
        self.spota_mu = spota_mu
        self.spota_alpha = spota_alpha
        self.ras_beta = ras_beta
        self.ras_tau = ras_tau
        self.assigner = TR3DAssigner(top_pts_threshold=32, label2level=[0])
        self.bbox_loss = AxisAlignedIoULoss2(mode='diou', reduction='none')
        self.cls_loss = FocalLoss(reduction='none')
        self.com_loss = FocalLoss(reduction='none')
        self.keep_loss = FocalLoss(reduction='mean', use_sigmoid=True)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_samples = (3200,320)
        self.num_samples_com = 2400
        self.com_threshold = com_threshold
        self.random_prune_threshold = (1200,4000)
        self._init_layers(in_channels, out_channels, n_reg_outs, n_classes)


    @staticmethod
    def make_block(in_channels, out_channels, kernel_size=3):
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels,
                                    kernel_size=kernel_size, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True))


    @staticmethod
    def make_down_block(in_channels, out_channels):
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3,
                                    stride=2, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True))


    @staticmethod
    def make_up_block(in_channels, out_channels, generative=False):
        conv = ME.MinkowskiGenerativeConvolutionTranspose if generative \
            else ME.MinkowskiConvolutionTranspose
        return nn.Sequential(
            conv(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True))


    def _init_layers(self, in_channels, out_channels, n_reg_outs, n_classes):
        self.bbox_conv = ME.MinkowskiConvolution(
            out_channels, n_reg_outs, kernel_size=1, bias=True, dimension=3)
        self.cls_conv = ME.MinkowskiConvolution(
            out_channels, n_classes, kernel_size=1, bias=True, dimension=3)
        self.keep_conv = nn.ModuleList([
            ME.MinkowskiConvolution(out_channels, 1, kernel_size=1, bias=True, dimension=3),
            ME.MinkowskiConvolution(out_channels, 1, kernel_size=1, bias=True, dimension=3)
        ])
        self.pos_embed = PositionEmbeddingLearned(3, 128)
        bi_layer0 = BiEncoderLayer(
            128, dropout=0.1, activation="relu",
            n_heads=8, dim_feedforward=128,
            self_attend_lang=True, self_attend_vis=True,
            use_butd_enc_attn=False
        )
        bi_layer1 = BiEncoderLayer(
            128, dropout=0.1, activation="relu",
            n_heads=8, dim_feedforward=128,
            self_attend_lang=True, self_attend_vis=True,
            use_butd_enc_attn=False
        )
        bi_layer2 = BiEncoderLayer(
            128, dropout=0.1, activation="relu",
            n_heads=8, dim_feedforward=128,
            self_attend_lang=True, self_attend_vis=True,
            use_butd_enc_attn=False
        )
        self.keep_trans = nn.ModuleList([BiEncoder(bi_layer0, 2), BiEncoder(bi_layer1, 2)])
        self.com_trans = BiEncoder(bi_layer2, 2)
        self.pruning = ME.MinkowskiPruning()
        self.com_cls = nn.Conv1d(128, 1, kernel_size=1, bias=True)


        for i in range(len(in_channels)):
            if i > 0:
                self.__setattr__(
                    f'up_block_{i}',
                    self.make_up_block(in_channels[i], in_channels[i - 1], generative=True))
            self.__setattr__(
                        f'lateral_block_{i}',
                        self.make_block(in_channels[i], in_channels[i]))
            if i == 0:
                self.__setattr__(
                    f'out_block_{i}',
                    self.make_block(in_channels[i], out_channels))

        self.fuse = MinkowskiFeatureFusionBlock(128, 128, 128)


    def init_weights(self):
        nn.init.normal_(self.bbox_conv.kernel, std=.01)
        nn.init.normal_(self.cls_conv.kernel, std=.01)
        nn.init.constant_(self.cls_conv.bias, bias_init_with_prob(.01))

        for i in range(len(self.keep_conv)):
            nn.init.normal_(self.keep_conv[i].kernel, std=.01)

        for n, m in self.named_modules():
            if ('bbox_conv' not in n) and ('cls_conv' not in n) \
                and ('keep_conv' not in n) and ('loss' not in n):
                if isinstance(m, ME.MinkowskiConvolution):
                    ME.utils.kaiming_normal_(
                        m.kernel, mode='fan_out', nonlinearity='relu')

                if isinstance(m, ME.MinkowskiBatchNorm):
                    nn.init.constant_(m.bn.weight, 1)
                    nn.init.constant_(m.bn.bias, 0)       
    

    def _forward_single(self, x):
        #Step 25-1: Bbox 오프셋·크기 회귀 — bbox_conv(MinkowskiConv1×1)로 각 voxel에서 6개 파라미터 예측
        # out(N,128) → reg_final(N,6): [Δx, Δy, Δz, log_w, log_h, log_d]
        # Δxyz: voxel 중심에서 박스 중심까지의 오프셋(m). log_whd: 박스 크기의 log scale 값.
        reg_final = self.bbox_conv(x).features

        #Step 25-2: 박스 크기 활성화 — log scale 크기를 exp()로 변환해 반드시 양수(m)로 보장
        # reg_final[:,3:6](N,3) log_whd → reg_distance(N,3) whd (양수, 단위: m)
        # exp()를 쓰는 이유: 예측값이 음수가 되더라도 박스 크기는 항상 양수가 되도록 강제.
        reg_distance = torch.exp(reg_final[:, 3:6])
        reg_angle = reg_final[:, 6:]

        #Step 25-3: Bbox 파라미터 조합 — 오프셋(3) + 크기(3)를 이어붙여 최종 bbox 파라미터 생성
        # reg_final[:,:3](N,3) + reg_distance(N,3) → bbox_pred(N,6): [Δx,Δy,Δz, w,h,d]
        # 실제 박스 중심 = voxel 좌표 + Δxyz. _bbox_pred_to_bbox()에서 최종 좌표로 변환.
        bbox_pred = torch.cat((reg_final[:, :3], reg_distance, reg_angle), dim=1)

        #Step 25-4: Foreground 스코어 예측 — cls_conv(MinkowskiConv1×1)로 물체 존재 확률 예측
        # out(N,128) → cls_pred(N,1): 각 voxel이 물체 중심 근방인지 여부의 이진 점수
        # 학습 시 TR3DAssigner가 positive/negative를 결정 → FocalLoss로 지도.
        scores = self.cls_conv(x)
        cls_pred = scores.features

        #Step 25-5: 배치 분리 + voxel 좌표 실수 변환 — 배치별 voxel 목록으로 분리 후 좌표를 m 단위로 환산
        # bbox_pred(N,6), cls_pred(N,1), x.coordinates(N,4) → list[B](N_i,6), list[B](N_i,1), list[B](N_i,3)
        # decomposition_permutations: SparseTensor 내 배치별 인덱스. coordinates[:,1:]×voxel_size로 정수 격자→미터 변환.
        bbox_preds, cls_preds, points = [], [], []
        for permutation in x.decomposition_permutations:
            bbox_preds.append(bbox_pred[permutation])
            cls_preds.append(cls_pred[permutation])
            points.append(x.coordinates[permutation][:, 1:]* self.voxel_size)
        return bbox_preds, cls_preds, points


    def forward(self, x,text_feats, text_attention_mask, gt_bboxes, gt_labels, gt_all_bbox_new, auxi_bbox, img_metas,pc=None):
        #Step 9: GT Bbox 레벨 분류 — 각 GT 박스를 볼륨 기준으로 레벨(0=small, 1=large)로 태깅
        # gt_bboxes, gt_all_bbox_new, auxi_bbox → bboxes_state: list[B] Tensor(N_bbox, 8) [level, cx,cy,cz, w,h,d, ...]
        # bboxes_state의 첫 번째 열이 레벨 인덱스. 이후 _get_keep_voxel에서 레벨별로 필터링하여 GT keep mask 생성.
        bboxes_level = []
        bboxes_state = []
        if self.assign_type == 'volume':
            for idx in range(len(img_metas)):
 
                bbox_all = gt_all_bbox_new[idx]
                bbox_level = torch.ones([bbox_all.shape[0], 1])
                bbox_state_all = torch.cat((bbox_level, bbox_all.gravity_center, bbox_all.tensor[:, 3:]), dim=1)

                bbox_gt = gt_bboxes[idx]
                bbox_state_gt = torch.cat((bbox_gt.gravity_center, bbox_gt.tensor[:, 3:]), dim=1)                
                bbox_auxi = auxi_bbox[idx]
                bbox_state_auxi = torch.cat((bbox_auxi.gravity_center, bbox_auxi.tensor[:, 3:]), dim=1)
                bbox_state_auxi_gt = torch.cat((bbox_state_gt, bbox_state_auxi), dim=0)
                bbox_level = torch.zeros([bbox_state_auxi_gt.shape[0], 1])
                bbox_state_auxi_gt = torch.cat((bbox_level, bbox_state_auxi_gt), dim=1)
                
                bbox_state = torch.cat((bbox_state_all, bbox_state_auxi_gt), dim=0)
                
                bboxes_level.append(bbox_state[:,[0]])
                bboxes_state.append(bbox_state)
        
        bbox_preds, cls_preds, points = [], [], []
        keep_gts = []
        keep_preds, prune_masks = [], []
        prune_mask = None
        inputs = x[1:]  # backbone 출력 list[4]에서 layer1 제외: inputs=[layer2, layer3, layer4]
        x = inputs[-1]  # 가장 거친 레벨(layer4)에서 시작
        # 루프 구조: i=2(레벨2, 가장 거침) → i=1(레벨1) → i=0(레벨0, 가장 섬세)
        # if i==1: GT keep mask 계산 + upsample + 레벨1 특징 합산 + Step14 pruning
        # elif i==0: GT keep mask 계산 + upsample + Step19 pruning + Completion branch(Step 20~22)
        # if i>0(공통): 복셀 샘플링 + BiEncoder + keep_conv (Step 10~11 / Step 15~16)
        for i in range(len(inputs) - 1, -1, -1): # 2,1,0
            if i ==1 :  #  1,0
                #Step 13: [레벨1] Keep Voxel GT 계산 + Upsample + 레벨1 특징 합산
                # x(레벨2 SparseTensor) → prune_mask: (N_voxels_level2,) bool, keep_gt: list[B] bool
                # _get_keep_voxel: 각 voxel이 GT bbox 영역 안에 있으면 True. 레벨2→1 upsample 후 레벨1 features 합산.
                prune_mask = self._get_keep_voxel(x, i + 2, bboxes_state, img_metas)

                keep_gt = []
                for permutation in x.decomposition_permutations:
                    keep_gt.append(prune_mask[permutation])
                keep_gts.append(keep_gt)
                x = self.__getattr__(f'up_block_{i + 1}')(x)
                coords = x.coordinates.float()
                x_level_features = inputs[i].features_at_coordinates(coords)  # select for partial addition
                x_level = ME.SparseTensor(features=x_level_features,
                                          coordinate_map_key=x.coordinate_map_key,
                                        coordinate_manager=x.coordinate_manager)
                x = x + x_level

                #Step 14: [레벨1] 레벨2 Keep Score 기반 Pruning — keep_conv[1] 예측값으로 불필요 voxel 제거
                # x(N_voxels_level1,128) → x(N_pruned_level1,128), prune_training_keep는 이전 i=2 순회에서 계산됨
                # TopK 방식: pts_prune_threshold[1]=4000개 voxel만 유지. GT 박스 주변 voxel을 선별적으로 보존.
                x = self._prune_training(x, prune_training_keep, i)
            elif i == 0:
                #Step 18: [레벨0] Keep Voxel GT 계산 + Upsample + 레벨1 Keep Score 기반 Pruning
                # x(레벨1 SparseTensor) → GT keep mask 계산 → 레벨1→0 upsample → random TopK pruning(1200~4000개)
                # random_prune_threshold 범위에서 임의 수를 선택해 데이터 증강 효과. prune_training_keep는 i=1 순회에서 계산됨.
                prune_mask = self._get_keep_voxel(x, i + 2, bboxes_state, img_metas)
                keep_gt = []
                for permutation in x.decomposition_permutations:
                    keep_gt.append(prune_mask[permutation])
                keep_gts.append(keep_gt)
                x = self.__getattr__(f'up_block_{i + 1}')(x)
                prune_threshold_ = np.random.randint(self.random_prune_threshold[0], self.random_prune_threshold[1])
                self.pts_prune_threshold = (prune_threshold_,self.pts_prune_threshold[1])
                x = self._prune_training(x, prune_training_keep, i)

                #Step 19: [레벨0] 레벨0(layer1) 특징 합산 → x_ori 생성
                # x(N_pruned,128) + inputs[0] features → x_ori: SparseTensor(N_pruned,128)
                # inputs[0](layer2 output)의 fine-grained 특징을 pruned voxel 위치에서 추출해 더함.
                coords = x.coordinates.float()
                x_level_features = inputs[i].features_at_coordinates(coords)  # select for partial addition
                x_level = ME.SparseTensor(features=x_level_features,
                                          coordinate_map_key=x.coordinate_map_key,
                                        coordinate_manager=x.coordinate_manager)
                x_ori = x + x_level

                #Step 20: [Completion] 원본 layer1 포인트 샘플링 + com_trans BiEncoder — 누락 voxel 보완
                # inputs[0](N_all_voxels,64) → sampled_features: (B,2400,128), sampled_coords: (B,2400,4)
                # 2400개 미만이면 zero-padding. com_trans(BiEncoder×2): 텍스트와 교차 주의하여 물체 관련 voxel 활성화.
                sampled_coords,sampled_features, original_indices = [],[],[]

                for permutation in inputs[0].decomposition_permutations:
                    original_indices.extend(permutation.cpu().numpy())
                    if len(permutation) > self.num_samples_com:
                        choice = torch.randperm(len(permutation))[:self.num_samples_com]
                        choice = torch.sort(choice).values
                        sampled_features.append(inputs[0].features[permutation][choice])
                        sampled_coords.append(inputs[0].coordinates[permutation][choice])
                    else:
                        padding_size = self.num_samples_com - len(permutation)
                        padded_features = torch.cat(
                            [inputs[0].features[permutation], torch.zeros((padding_size, inputs[0].features[permutation].shape[1]),
                                                                  dtype=inputs[0].features.dtype).to(inputs[0].device)], dim=0)
                        padded_coords = torch.cat(
                            [inputs[0].coordinates[permutation], -torch.ones((padding_size, inputs[0].coordinates[permutation].shape[1]),
                                                                     dtype=inputs[0].coordinates.dtype).to(inputs[0].device)],
                                                                     dim=0)
                        sampled_features.append(padded_features)
                        sampled_coords.append(padded_coords)
                sampled_features = torch.stack(sampled_features)
                sampled_coords = torch.stack(sampled_coords)
                sampled_features, text_feats = self.com_trans(
                    vis_feats=sampled_features.contiguous(),
                    pos_feats=self.pos_embed(sampled_coords[:,:,1:]*self.voxel_size).transpose(1, 2).contiguous(),
                    padding_mask=sampled_coords[:, :,0] == -1,
                    text_feats=text_feats,
                    text_padding_mask=text_attention_mask)

                #Step 21: [Completion] com_cls 예측 + 임계값 필터링 + x_ori 중복 제거
                # sampled_features: (B,2400,128) → com_pred: (B,2400,1) → sigmoid > 0.15인 voxel만 보존
                # com_pred_training: 손실 계산용 보존. sigmoid 임계값(=0.15)으로 물체 관련 voxel 선별.
                # x_ori에 이미 있는 좌표는 제거(matches)하여 중복 추가를 방지.
                com_pred = self.com_cls(sampled_features.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
                valid_mask = sampled_coords[:, :,0] != -1
                com_pred_training = [com_pred[k][valid_mask[k]] for k in range(len(com_pred))]
                com_coords_training = [sampled_coords[k][valid_mask[k]][:,1:]*self.voxel_size for k in range(len(com_pred))]
                sampled_features = sampled_features[valid_mask]
                sampled_coords = sampled_coords[valid_mask]
                com_pred = com_pred[valid_mask].squeeze(-1)
                com_mask = com_pred.sigmoid() > self.com_threshold
                sampled_features = sampled_features[com_mask]
                sampled_coords = sampled_coords[com_mask]
                matches = (sampled_coords.unsqueeze(1) == x_ori.coordinates.unsqueeze(0)).all(dim=-1).any(dim=1)
                sampled_features = sampled_features[~matches]
                sampled_coords = sampled_coords[~matches]

                #Step 22: [Completion] Completion Voxel 병합 — x_ori와 com voxel을 합쳐 최종 레벨0 SparseTensor 생성
                # x_ori(N_pruned,128) + com_voxels(N_com,128) → x: SparseTensor(N_pruned+N_com, 128)
                # x_com_features: x_ori에서 com 좌표의 특징을 보간 후 com_features와 합산하여 풍부한 표현 생성.
                x_com_features = x.features_at_coordinates(sampled_coords.float())
                x_com_features = x_com_features + sampled_features
                x = ME.SparseTensor(features=torch.cat((x_ori.features,x_com_features),dim=0),
                                    coordinates=torch.cat((x_ori.coordinates,sampled_coords),dim=0),
                                    coordinate_manager=x_ori.coordinate_manager, tensor_stride=x_ori.tensor_stride, device=x_ori.device)
            if i > 0: # 2,1
                #Step 10 (i=2, 레벨2) / Step 15 (i=1, 레벨1): Voxel 균일 샘플링 + BiEncoder (keep_trans[i-1])
                # x(N_voxels,128) → sampled_features: (B, num_samples[i-1], 128), sampled_coords: (B, num_samples[i-1], 4)
                # i=2: 3200개, i=1: 320개 샘플링. 부족하면 zero-padding(-1 좌표). keep_trans: 텍스트와 교차 주의하여 언어-유도 특징 갱신.
                sampled_coords,sampled_features, original_indices = [],[],[]
                prune_mask = torch.zeros(x.shape[0], dtype=torch.bool).to(x.device)
                for permutation in x.decomposition_permutations:
                    original_indices.extend(permutation.cpu().numpy())
                    if len(permutation) > self.num_samples[i-1]:
                        choice = torch.randperm(len(permutation))[:self.num_samples[i-1]]
                        choice = torch.sort(choice).values
                        sampled_features.append(x.features[permutation][choice])
                        sampled_coords.append(x.coordinates[permutation][choice])
                        prune_mask[permutation[choice]] = True
                    else:
                        padding_size = self.num_samples[i-1] - len(permutation)
                        padded_features = torch.cat(
                            [x.features[permutation], torch.zeros((padding_size, x.features[permutation].shape[1]),
                                                                  dtype=x.features.dtype).to(x.device)], dim=0)
                        padded_coords = torch.cat(
                            [x.coordinates[permutation], -torch.ones((padding_size, x.coordinates[permutation].shape[1]),
                                                                     dtype=x.coordinates.dtype).to(x.device)],
                                                                     dim=0)
                        sampled_features.append(padded_features)
                        sampled_coords.append(padded_coords)
                        prune_mask[permutation] = True
                sampled_features = torch.stack(sampled_features)
                sampled_coords = torch.stack(sampled_coords)
                sampled_features, text_feats = self.keep_trans[i-1](
                    vis_feats=sampled_features.contiguous(),
                    pos_feats=self.pos_embed(sampled_coords[:,:,1:]*self.voxel_size).transpose(1, 2).contiguous(),
                    padding_mask=sampled_coords[:, :,0] == -1,
                    text_feats=text_feats,
                    text_padding_mask=text_attention_mask)

                valid_mask = sampled_coords[:, :,0] != -1
                sampled_features = sampled_features[valid_mask]
                sampled_coords = sampled_coords[valid_mask]

                x = ME.SparseTensor(features=sampled_features, coordinates=sampled_coords,
                                    coordinate_manager=x.coordinate_manager, tensor_stride=x.tensor_stride, device=x.device)

                #Step 11 (i=2, 레벨2) / Step 16 (i=1, 레벨1): Keep Score 예측 — keep_conv[i-1]로 각 voxel의 보존 확률 예측
                # x(N_sampled,128) → keep_scores: SparseTensor(N_sampled,1), prune_training_keep: SparseTensor(negative scores)
                # keep_conv[i-1]: 1×1 MinkowskiConv. prune_training_keep(-keep_scores)는 다음 레벨 pruning에 사용됨.
                keep_scores = self.keep_conv[i-1](x) # 1 MLP
                prune_training_keep = ME.SparseTensor(
                                    -keep_scores.features,
                                    coordinate_map_key=keep_scores.coordinate_map_key,
                                    coordinate_manager=keep_scores.coordinate_manager)
                
     
                keep_pred = keep_scores.features
                prune_inference = keep_pred
                keeps = []

                try:
                    for permutation in x.decomposition_permutations:
                        keeps.append(keep_pred[permutation])
                except:
                    pdb.set_trace()
                keep_preds.append(keeps)
                
            #Step 12 (i=2) / Step 17 (i=1) / Step 23 (i=0): Lateral Block 처리 — 채널 정규화 및 특징 정제
            # x(N,128) → x(N,128); Conv3×3→BN→ReLU
            # 다음 레벨로 넘기기 전 특징을 안정화.
            x = self.__getattr__(f'lateral_block_{i}')(x)
            if i == 0:
                #Step 23 (계속): Out Block 처리 — 최종 레벨0 특징을 출력 채널(128)로 변환
                # x(N_level0,128) → out: SparseTensor(N_level0,128)
                out = self.__getattr__(f'out_block_{i}')(x)

        #Step 24: Text-Visual Fusion — out voxel 특징에 텍스트 [CLS] 토큰을 채널 방향으로 융합
        # out(N_level0,128) + text_feats[:,0](B,128) → out: SparseTensor(N_level0,128)
        # MinkowskiFeatureFusionBlock: batch 인덱스별로 [CLS] 토큰을 반복 확장 후 concat → Conv1×1→BN→ReLU.
        out = self.fuse(out, text_feats[:, 0])

        #Step 25-1~25-5: 최종 Bbox/Cls 예측 — _forward_single(out)로 각 voxel에서 박스·스코어 예측 후 배치 분리
        # out(N_level0,128) → bbox_pred: list[B](N_i,6), cls_pred: list[B](N_i,1), point: list[B](N_i,3)
        # Step 25-1: bbox_conv → (N,6) [Δx,Δy,Δz, log_w,log_h,log_d]
        # Step 25-2: exp(log_whd) → 양수 크기(m) 보장
        # Step 25-3: offset + size concat → bbox_pred(N,6)
        # Step 25-4: cls_conv → foreground score(N,1)
        # Step 25-5: decomposition_permutations로 배치 분리 + coordinates×voxel_size로 m 단위 좌표 변환
        bbox_pred, cls_pred, point = self._forward_single(out)
        return [bbox_pred], [cls_pred], [point], keep_preds[::-1], keep_gts[::-1], bboxes_level, com_pred_training, com_coords_training
    

    def _prune_inference(self, x, scores, layer_id):
        """Prunes the tensor by score thresholding.

        Args:
            x (SparseTensor): Tensor to be pruned.
            scores (SparseTensor): Scores for thresholding.

        Returns:
            SparseTensor: Pruned tensor.
        """
        with torch.no_grad():
            prune_mask = scores.new_zeros(
                (len(scores)), dtype=torch.bool)

            for permutation in x.decomposition_permutations:
                score = scores[permutation].sigmoid()
                score = 1 - score
                mask = score > self.prune_threshold[layer_id]
                mask = mask.reshape([len(score)])
                prune_mask[permutation[mask]] = True                 
        if prune_mask.sum() != 0:
            x = self.pruning(x, prune_mask)
        else:
            x = None

        return x


    def _prune_training(self, x, scores, layer_id):
        """Prunes the tensor by score thresholding.

        Args:
            x (SparseTensor): Tensor to be pruned.
            scores (SparseTensor): Scores for thresholding.

        Returns:
            SparseTensor: Pruned tensor.
        """

        with torch.no_grad():
            coordinates = x.C.float()
            interpolated_scores = scores.features_at_coordinates(coordinates)
            prune_mask = interpolated_scores.new_zeros(
                (len(interpolated_scores)), dtype=torch.bool)
            for permutation in x.decomposition_permutations:
                score = interpolated_scores[permutation]
                mask = score.new_zeros((len(score)), dtype=torch.bool)
                topk = min(len(score), self.pts_prune_threshold[layer_id])
                ids = torch.topk(score.squeeze(1), topk, sorted=False).indices
                mask[ids] = True
                prune_mask[permutation[mask]] = True
        x = self.pruning(x, prune_mask)
        return x


    @torch.no_grad()
    def _get_keep_voxel(self, input, cur_level, bboxes_state, input_metas):
        bboxes = []
        for size in range(len(input_metas)):
            bboxes.append([])
        for idx in range(len(input_metas)):
            for n in range(len(bboxes_state[idx])):
                if bboxes_state[idx][n][0] < (cur_level - 1):    
                    bboxes[idx].append(bboxes_state[idx][n])
        idx = 0
        mask = []
        l0 = self.voxel_size * 2 ** 2  # pool  True :2**3  False:2**2
        for idx, permutation in enumerate(input.decomposition_permutations):
            point = input.coordinates[permutation][:, 1:]* self.voxel_size
            if len(bboxes[idx]) != 0:
                point = input.coordinates[permutation][:, 1:]* self.voxel_size
                boxes = bboxes[idx]
                level = 3
                bboxes_level = [[] for _ in range(level)]
                for n in range(len(boxes)):
                    for l in range(level):
                        if boxes[n][0] == l:
                            bboxes_level[l].append(boxes[n])
                inside_box_conditions = torch.zeros((len(permutation)), dtype=torch.bool).to(point.device)
                for l in range(level):
                    if len(bboxes_level[l]) != 0:
                        point_l = point.unsqueeze(1).expand(len(point), len(bboxes_level[l]), 3)
                        boxes_l = torch.cat(bboxes_level[l]).reshape([-1, 8]).to(point.device)
                        boxes_l = boxes_l.expand(len(point), len(bboxes_level[l]), 8)
                        shift = torch.stack(
                            (point_l[..., 0] - boxes_l[..., 1], point_l[..., 1] - boxes_l[..., 2],
                            point_l[..., 2] - boxes_l[..., 3]),
                            dim=-1).permute(1, 0, 2)
                        shift = rotation_3d_in_axis(
                            shift, -boxes_l[0, :, 7], axis=2).permute(1, 0, 2)
                        centers = boxes_l[..., 1:4] + shift
                        up_level_l = self.r[cur_level-2] 
                        dx_min = centers[..., 0] - boxes_l[..., 1] + (up_level_l * l0 * 2 ** (cur_level - 1)) / 2  
                        dx_max = boxes_l[..., 1] - centers[..., 0] + (up_level_l * l0 * 2 ** (cur_level - 1)) / 2 
                        dy_min = centers[..., 1] - boxes_l[..., 2] + (up_level_l * l0 * 2 ** (cur_level - 1)) / 2  
                        dy_max = boxes_l[..., 2] - centers[..., 1] + (up_level_l * l0 * 2 ** (cur_level - 1)) / 2
                        dz_min = centers[..., 2] - boxes_l[..., 3] + (up_level_l * l0 * 2 ** (cur_level - 1)) / 2  
                        dz_max = boxes_l[..., 3] - centers[..., 2] + (up_level_l * l0 * 2 ** (cur_level - 1)) / 2


                        distance = torch.stack((dx_min, dx_max, dy_min, dy_max, dz_min, dz_max), dim=-1)
                        inside_box_condition = distance.min(dim=-1).values > 0
                        inside_box_condition = inside_box_condition.sum(dim=1)
                        inside_box_condition = inside_box_condition >= 1
                        inside_box_conditions += inside_box_condition
                mask.append(inside_box_conditions)
            else:
                inside_box_conditions = torch.zeros((len(permutation)), dtype=torch.bool).to(point.device)
                mask.append(inside_box_conditions)

        prune_mask = torch.cat(mask)
        prune_mask = prune_mask.to(input.device)
        return prune_mask
    

    @staticmethod
    def _bbox_to_loss(bbox):
        """Transform box to the axis-aligned or rotated iou loss format.
        Args:
            bbox (Tensor): 3D box of shape (N, 6) or (N, 7).
        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        # rotated iou loss accepts (x, y, z, w, h, l, heading)
        if bbox.shape[-1] != 6:
            return bbox

        # axis-aligned case: x, y, z, w, h, l -> x1, y1, z1, x2, y2, z2
        return torch.stack(
            (bbox[..., 0] - bbox[..., 3] / 2, bbox[..., 1] - bbox[..., 4] / 2,
             bbox[..., 2] - bbox[..., 5] / 2, bbox[..., 0] + bbox[..., 3] / 2,
             bbox[..., 1] + bbox[..., 4] / 2, bbox[..., 2] + bbox[..., 5] / 2),
            dim=-1)


    @staticmethod
    def _bbox_pred_to_bbox(points, bbox_pred):
        """Transform predicted bbox parameters to bbox.
        Args:
            points (Tensor): Final locations of shape (N, 3)
            bbox_pred (Tensor): Predicted bbox parameters of shape (N, 6)
                or (N, 8).
        Returns:
            Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
        """
        if bbox_pred.shape[0] == 0:
            return bbox_pred

        x_center = points[:, 0] + bbox_pred[:, 0]
        y_center = points[:, 1] + bbox_pred[:, 1]
        z_center = points[:, 2] + bbox_pred[:, 2]
        base_bbox = torch.stack([
            x_center,
            y_center,
            z_center,
            bbox_pred[:, 3],
            bbox_pred[:, 4],
            bbox_pred[:, 5]], -1)

        # axis-aligned case
        if bbox_pred.shape[1] == 6:
            return base_bbox

        # rotated case: ..., sin(2a)ln(q), cos(2a)ln(q)
        scale = bbox_pred[:, 3] + bbox_pred[:, 4]
        q = torch.exp(
            torch.sqrt(
                torch.pow(bbox_pred[:, 6], 2) + torch.pow(bbox_pred[:, 7], 2)))
        alpha = 0.5 * torch.atan2(bbox_pred[:, 6], bbox_pred[:, 7])
        return torch.stack(
            (x_center, y_center, z_center, scale / (1 + q), scale /
             (1 + q) * q, bbox_pred[:, 5] + bbox_pred[:, 4], alpha),
            dim=-1)


    def _loss_single(self,
                     bbox_preds,
                     cls_preds,
                     points,
                     gt_bboxes,
                     gt_labels,
                     img_meta,
                     com_pred,com_coords):
        bbox_preds_cat = torch.cat(bbox_preds)
        points_cat = torch.cat(points)

        # 메인 voxel 배정 — SPOTA on이면 예측 박스 기반 cost로, off면 기존 center-distance로.
        if self.use_spota:
            pred_boxes = self._bbox_pred_to_bbox(
                points_cat, bbox_preds_cat).detach()
            assigned_ids = self.assigner.assign(
                points, gt_bboxes, gt_labels, img_meta,
                bbox_preds=pred_boxes, use_spota=True,
                k=self.spota_k, mu=self.spota_mu, alpha=self.spota_alpha,
                cls_preds=torch.cat(cls_preds).detach())
        else:
            assigned_ids = self.assigner.assign(points, gt_bboxes, gt_labels, img_meta)

        bbox_preds = bbox_preds_cat
        cls_preds = torch.cat(cls_preds)
        points = points_cat

        # cls loss
        n_classes = cls_preds.shape[1]
        pos_mask = assigned_ids >= 0

        if len(gt_labels) > 0:
            cls_targets = torch.where(pos_mask, gt_labels[assigned_ids], n_classes)
        else:
            cls_targets = gt_labels.new_full((len(pos_mask),), n_classes)

        if self.use_ras:
            cls_loss = self._ras_cls_loss(
                cls_preds, cls_targets, pos_mask, bbox_preds, points,
                assigned_ids, gt_bboxes)
        else:
            cls_loss = self.cls_loss(cls_preds, cls_targets)

        # completion voxel 배정 — SPOTA/RAS 대상 아님(예측 박스가 없음), 기존 center-distance 그대로.
        assigned_ids_com = self.assigner.assign([com_coords], gt_bboxes, gt_labels, img_meta)
        # cls loss
        pos_mask_com = assigned_ids_com >= 0

        if len(gt_labels) > 0:
            cls_targets = torch.where(pos_mask_com, gt_labels[assigned_ids_com], n_classes)
        else:
            cls_targets = gt_labels.new_full((len(pos_mask_com),), n_classes)

        com_loss = self.com_loss(com_pred, cls_targets)

        # bbox loss
        pos_bbox_preds = bbox_preds[pos_mask]
        if pos_mask.sum() > 0:
            pos_points = points[pos_mask]
            pos_bbox_preds = bbox_preds[pos_mask]
            bbox_targets = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
            pos_bbox_targets = bbox_targets.to(points.device)[assigned_ids][pos_mask]
            if pos_bbox_preds.shape[1] == 6:
                pos_bbox_targets = pos_bbox_targets[:, :6]
            
            bbox_loss = self.bbox_loss(
                self._bbox_to_loss(self._bbox_pred_to_bbox(pos_points, pos_bbox_preds)),
                self._bbox_to_loss(pos_bbox_targets))            
        else:
            bbox_loss = None
        return bbox_loss, cls_loss, pos_mask, com_loss, pos_mask_com


    def _ras_cls_loss(self, cls_preds, cls_targets, pos_mask, bbox_preds,
                      points, assigned_ids, gt_bboxes):
        """RAS(Rank-aware Adaptive Self-Distillation) 분류 손실. SR3D Eq.6-7.
        positive voxel만 FocalLoss<->RDL(self-distillation) 적응 혼합, negative는 FocalLoss 그대로.
        """
        fl = self.cls_loss(cls_preds, cls_targets)  # (N, n_classes), 기존과 동일한 FocalLoss
        if pos_mask.sum() == 0:
            return fl

        pos_points = points[pos_mask]
        pos_bbox_preds = bbox_preds[pos_mask]
        pos_pred_boxes = self._bbox_pred_to_bbox(pos_points, pos_bbox_preds)

        bbox_targets = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
        pos_gt_boxes = bbox_targets.to(points.device)[assigned_ids][pos_mask]
        if pos_pred_boxes.shape[1] == 6:
            pos_gt_boxes = pos_gt_boxes[:, :6]

        # q_i = localization quality (IoU)
        q = axis_aligned_bbox_overlaps_3d(
            self._bbox_to_loss(pos_pred_boxes), self._bbox_to_loss(pos_gt_boxes),
            mode='iou', is_aligned=True
        ).clamp(min=1e-6, max=1 - 1e-6)

        # sig_i = 분류 confidence
        sig = cls_preds[pos_mask].sigmoid().squeeze(-1).clamp(min=1e-6, max=1 - 1e-6)

        r_reg = soft_rank(q, tau=self.ras_tau)     # IoU 순위 (잘 맞출수록 1에 가까움)
        r_cls = soft_rank(sig, tau=self.ras_tau)   # confidence 순위

        # RDL: self-distillation 손실 (SR3D Eq.6), 부호는 cross-entropy 관례에 맞춰 -RDL을 손실로 사용.
        rdl = (1 - r_reg).pow(self.ras_beta) * (q * torch.log(sig)) \
            + q * (1 - q) * torch.log(1 - sig)
        rdl_loss = -rdl

        fl_pos = fl[pos_mask].squeeze(-1)
        mixed_pos = (1 - r_cls) * fl_pos + r_cls * rdl_loss

        cls_loss = fl.clone()
        cls_loss[pos_mask] = mixed_pos.unsqueeze(-1)
        return cls_loss


    def _loss(self, bbox_preds, cls_preds, points, gt_bboxes, gt_labels, img_metas,
              keep_preds, keep_gts, bboxes_level, com_pred_training, com_coords_training):
        bbox_losses, cls_losses, pos_masks, com_losses, pos_masks_com = [], [], [], [], []

        #Step 26: Keep Loss 계산 — 레벨2·1의 voxel 보존 예측(keep_pred)과 GT 마스크(keep_gt) 간 FocalLoss
        # keep_preds: list[2] of list[B](N_i,1), keep_gts: list[2] of list[B](N_i,) bool → keep_losses: scalar
        # 레벨당 loss를 /3으로 정규화 후 배치 평균. GT 박스 안에 있는 voxel=1, 밖=0으로 지도.
        keep_losses = 0
        for i in range(len(img_metas)):
            k_loss = 0
            keep_pred = [x[i] for x in keep_preds]
            keep_gt = [x[i] for x in keep_gts]
            for j in range(len(keep_preds)):
                pred = keep_pred[j]
                gt = (keep_gt[j]).long()

                if gt.sum() != 0:
                    keep_loss = self.keep_loss(pred, gt, avg_factor=gt.sum())
                    k_loss = torch.mean(keep_loss) / 3 + k_loss
                else:
                    keep_loss = self.keep_loss(pred, gt, avg_factor=len(gt))
                    k_loss = torch.mean(keep_loss) / 3 + k_loss

            keep_losses = keep_losses + k_loss

        #Step 27: Bbox / Cls / Com Loss 계산 — 샘플별 TR3DAssigner로 positive 할당 후 각 손실 집계
        # gt_bboxes, bbox_preds, cls_preds, com_pred_training → bbox_losses, cls_losses, com_losses: list[B]
        # bbox_loss: DiIoU(AxisAlignedIoULoss2). cls_loss: FocalLoss(물체 여부). com_loss: FocalLoss(completion).
        # positive: assigner.assign으로 GT 박스와 가장 가까운 top_pts_threshold=32개 voxel 지정.
        for i in range(len(img_metas)):
            bbox_loss, cls_loss, pos_mask, com_loss,pos_mask_com = self._loss_single(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                points=[x[i] for x in points],
                img_meta=img_metas[i],
                gt_bboxes=gt_bboxes[i],
                gt_labels=gt_labels[i],
                com_pred = com_pred_training[i],
                com_coords = com_coords_training[i])
            if bbox_loss is not None:
                bbox_losses.append(bbox_loss)
            cls_losses.append(cls_loss)
            com_losses.append(com_loss)
            pos_masks.append(pos_mask)
            pos_masks_com.append(pos_mask_com)

        #Step 28: 손실 집계 및 딕셔너리 반환 — 4개 손실을 배치 평균하여 반환
        # bbox_losses, cls_losses, keep_losses, com_losses → dict{bbox_loss, cls_loss, keep_loss, com_loss}
        # bdetr.py에서 총합 loss = bbox_loss + cls_loss + keep_loss + com_loss 계산. 각 손실에 weight 적용.
        return dict(
            bbox_loss=self.bbox_loss_weight * torch.mean(torch.cat(bbox_losses)),
            cls_loss=torch.sum(torch.cat(cls_losses)) / torch.sum(torch.cat(pos_masks)),
            keep_loss=self.keep_loss_weight * keep_losses / len(img_metas),
            com_loss=torch.sum(torch.cat(com_losses)) / torch.sum(torch.cat(pos_masks_com))) 


    def forward_train(self, x, text_feats, text_attention_mask, gt_bboxes, gt_labels, gt_all_bbox_new, auxi_bbox, img_metas,pc=None):
        bbox_preds, cls_preds, points, keep_preds, keep_gts, bboxes_level, com_pred_training, com_coords_training = \
            self(x, text_feats, text_attention_mask, gt_bboxes, gt_labels, gt_all_bbox_new, auxi_bbox, img_metas,pc)

        return self._loss(bbox_preds, cls_preds, points,
                          gt_bboxes, gt_labels, img_metas, keep_preds, keep_gts, bboxes_level,
                          com_pred_training, com_coords_training)


    def _nms(self, bboxes, scores, img_meta):
        """Multi-class nms for a single scene.
        Args:
            bboxes (Tensor): Predicted boxes of shape (N_boxes, 6) or
                (N_boxes, 7).
            scores (Tensor): Predicted scores of shape (N_boxes, N_classes).
            img_meta (dict): Scene meta data.
        Returns:
            Tensor: Predicted bboxes.
            Tensor: Predicted scores.
            Tensor: Predicted labels.
        """
        n_classes = scores.shape[1]
        yaw_flag = bboxes.shape[1] == 7
        nms_bboxes, nms_scores, nms_labels = [], [], []
        for i in range(n_classes):
            ids = scores[:, i] > self.test_cfg['score_thr']
            if not ids.any():
                continue

            class_scores = scores[ids, i]
            class_bboxes = bboxes[ids]
            if yaw_flag:
                nms_function = nms3d
            else:
                class_bboxes = torch.cat(
                    (class_bboxes, torch.zeros_like(class_bboxes[:, :1])),
                    dim=1)
                nms_function = nms3d_normal

            nms_ids = nms_function(class_bboxes, class_scores,
                                   self.test_cfg['iou_thr'])
            nms_bboxes.append(class_bboxes[nms_ids])
            nms_scores.append(class_scores[nms_ids])
            nms_labels.append(
                bboxes.new_full(
                    class_scores[nms_ids].shape, i, dtype=torch.long))

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0, ))
            nms_labels = bboxes.new_zeros((0, ))

        if yaw_flag:
            box_dim = 7
            with_yaw = True
        else:
            box_dim = 6
            with_yaw = False
            nms_bboxes = nms_bboxes[:, :6]
        nms_bboxes = img_meta['box_type_3d'](
            nms_bboxes,
            box_dim=box_dim,
            with_yaw=with_yaw,
            origin=(.5, .5, .5))

        return nms_bboxes, nms_scores, nms_labels


    def _get_bboxes_single(self, bbox_preds, cls_preds, points, img_meta):
        scores = torch.cat(cls_preds).sigmoid()
        bbox_preds = torch.cat(bbox_preds)
        points = torch.cat(points)
        max_scores, _ = scores.max(dim=1)

        if len(scores) > self.test_cfg['nms_pre'] > 0:
            _, ids = max_scores.topk(self.test_cfg['nms_pre'])
            bbox_preds = bbox_preds[ids]
            scores = scores[ids]
            points = points[ids]

        boxes = self._bbox_pred_to_bbox(points, bbox_preds)
        labels = boxes.new_zeros((1, ),dtype=int)
        boxes = img_meta['box_type_3d'](boxes, box_dim=6, with_yaw=False, origin=(.5, .5, .5))
        return boxes, scores, labels


    def _get_bboxes(self, bbox_preds, cls_preds, points, img_metas):
        results = []
        for i in range(len(img_metas)):
            result = self._get_bboxes_single(
                bbox_preds=[x[i] for x in bbox_preds],
                cls_preds=[x[i] for x in cls_preds],
                points=[x[i] for x in points],
                img_meta=img_metas[i])
            results.append(result)
        return results


    def forward_test(self, x, text_feats, text_attention_mask, img_metas, pc=None, gt_bboxes=None):
        inputs = x[1:]
        x = inputs[-1]
        bbox_preds, cls_preds, points = [], [], []
        keep_scores = None
        
        for i in range(len(inputs) - 1, -1, -1):
            if i ==1:
                x = self._prune_inference(x, prune_inference,i)
                
                if x != None:
                    x = self.__getattr__(f'up_block_{i + 1}')(x)
                    coords = x.coordinates.float()
                    x_level_features = inputs[i].features_at_coordinates(coords)
                    x_level = ME.SparseTensor(features=x_level_features,
                                              coordinate_map_key=x.coordinate_map_key,
                                              coordinate_manager=x.coordinate_manager)
                    x = x + x_level
                else:
                    pdb.set_trace()
                    break
            elif i ==0:
                x = self._prune_inference(x, prune_inference,i)
                
                if x != None:
                    x = self.__getattr__(f'up_block_{i + 1}')(x)
                    coords = x.coordinates.float()
                    x_level_features = inputs[i].features_at_coordinates(coords)
                    x_level = ME.SparseTensor(features=x_level_features,
                                              coordinate_map_key=x.coordinate_map_key,
                                              coordinate_manager=x.coordinate_manager)
                    x_ori = x + x_level
                else:
                    pdb.set_trace()
                    break
        
                sampled_coords,sampled_features, original_indices = [],[],[]
                
                for permutation in inputs[0].decomposition_permutations:
                    original_indices.extend(permutation.cpu().numpy())
                    if len(permutation) > self.num_samples_com:
                        choice = torch.randperm(len(permutation))[:self.num_samples_com]
                        choice = torch.sort(choice).values
                        sampled_features.append(inputs[0].features[permutation][choice])
                        sampled_coords.append(inputs[0].coordinates[permutation][choice])
                    else:
                        padding_size = self.num_samples_com - len(permutation)      
                        padded_features = torch.cat(
                            [inputs[0].features[permutation], torch.zeros((padding_size, inputs[0].features[permutation].shape[1]), 
                                                                  dtype=inputs[0].features.dtype).to(inputs[0].device)], dim=0) 
                        padded_coords = torch.cat(
                            [inputs[0].coordinates[permutation], -torch.ones((padding_size, inputs[0].coordinates[permutation].shape[1]),
                                                                     dtype=inputs[0].coordinates.dtype).to(inputs[0].device)], 
                                                                     dim=0)  
                        sampled_features.append(padded_features)
                        sampled_coords.append(padded_coords)
                sampled_features = torch.stack(sampled_features)
                sampled_coords = torch.stack(sampled_coords)
                sampled_features, text_feats = self.com_trans(
                    vis_feats=sampled_features.contiguous(),
                    pos_feats=self.pos_embed(sampled_coords[:,:,1:]*self.voxel_size).transpose(1, 2).contiguous(),
                    padding_mask=sampled_coords[:, :,0] == -1,
                    text_feats=text_feats,
                    text_padding_mask=text_attention_mask)
                
                com_pred = self.com_cls(sampled_features.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
                valid_mask = sampled_coords[:, :,0] != -1
                sampled_features = sampled_features[valid_mask]
                sampled_coords = sampled_coords[valid_mask]
                com_pred = com_pred[valid_mask].squeeze(-1)
                com_mask = com_pred.sigmoid() > self.com_threshold
                sampled_features = sampled_features[com_mask]
                sampled_coords = sampled_coords[com_mask]                
                matches = (sampled_coords.unsqueeze(1) == x_ori.coordinates.unsqueeze(0)).all(dim=-1).any(dim=1)
                sampled_features = sampled_features[~matches]
                sampled_coords = sampled_coords[~matches]                   
                
                x_com_features = x.features_at_coordinates(sampled_coords.float())     
                x_com_features = x_com_features + sampled_features           
                x = ME.SparseTensor(features=torch.cat((x_ori.features,x_com_features),dim=0), 
                                    coordinates=torch.cat((x_ori.coordinates,sampled_coords),dim=0), 
                                    coordinate_manager=x_ori.coordinate_manager, tensor_stride=x_ori.tensor_stride, device=x_ori.device)
                
            if i > 0:
                sampled_coords,sampled_features = [],[]
                len_x = []
                for permutation in x.decomposition_permutations:
                    len_x.append(len(x.coordinates[permutation]))
                max_len_x = int(torch.tensor(len_x).max())
                if len(len_x)>1:
                    for permutation in x.decomposition_permutations:
                        if len(permutation) > max_len_x:
                            choice = torch.randperm(len(permutation))[:max_len_x]
                            choice = torch.sort(choice).values
                            sampled_features.append(x.features[permutation][choice])
                            sampled_coords.append(x.coordinates[permutation][choice])
                        else:
                            padding_size = max_len_x - len(permutation)      
                            padded_features = torch.cat(
                                [x.features[permutation], torch.zeros((padding_size, x.features[permutation].shape[1]), 
                                                                    dtype=x.features.dtype).to(x.device)], dim=0) 
                            padded_coords = torch.cat(
                                [x.coordinates[permutation], -torch.ones((padding_size, x.coordinates[permutation].shape[1]),
                                                                        dtype=x.coordinates.dtype).to(x.device)], 
                                                                        dim=0)   
                            sampled_features.append(padded_features)
                            sampled_coords.append(padded_coords)
                else:
                    for permutation in x.decomposition_permutations:
                        sampled_features.append(x.features[permutation])
                        sampled_coords.append(x.coordinates[permutation])                        
                sampled_features = torch.stack(sampled_features)
                sampled_coords = torch.stack(sampled_coords)
                sampled_features, text_feats = self.keep_trans[i-1](
                    vis_feats=sampled_features.contiguous(),
                    pos_feats=self.pos_embed(sampled_coords[:,:,1:]*self.voxel_size).transpose(1, 2).contiguous(),
                    padding_mask=sampled_coords[:, :,0] == -1,
                    text_feats=text_feats,
                    text_padding_mask=text_attention_mask)
                
                valid_mask = sampled_coords[:, :,0] != -1
                sampled_features = sampled_features[valid_mask]
                sampled_coords = sampled_coords[valid_mask]
                x = ME.SparseTensor(features=sampled_features, coordinates=sampled_coords, 
                                    coordinate_manager=x.coordinate_manager, tensor_stride=x.tensor_stride, device=x.device)
                keep_scores = self.keep_conv[i-1](x)
                keep_pred = keep_scores.features
                prune_inference = keep_pred

            x = self.__getattr__(f'lateral_block_{i}')(x)
            if i == 0:
                out = self.__getattr__(f'out_block_{i}')(x)
        start_time = time.time()
        out = self.fuse(out, text_feats[:, 0])
        bbox_pred, cls_pred, point = self._forward_single(out)
        results = self._get_bboxes([bbox_pred], [cls_pred], [point], img_metas)
        head_time = time.time() - start_time
        return results, head_time

def _spota_cost(points, boxes, bbox_preds, mu, alpha, cls_preds=None):
    """SPOTA(Spatial-Prioritized OTA) cost. SR3D 본문 Eq.3-5.
    points: (n_points, n_boxes, 3) 브로드캐스트된 voxel 좌표 (사용 안 함, box와 pred만 필요)
    boxes:  (n_points, n_boxes, 7) 브로드캐스트된 GT 박스 (마지막 열은 yaw/pad, axis-aligned에선 무시)
    bbox_preds: (n_points, 6) voxel별 예측 박스 [cx,cy,cz,w,h,d]
    반환: (n_points, n_boxes) cost 행렬. 작을수록 좋은(positive에 가까운) 후보.
    """
    if bbox_preds.shape[-1] != 6:
        raise NotImplementedError(
            'SPOTA assign only supports axis-aligned (6-dim) box predictions; '
            'rotated (NR3D/SR3D, box_dim=7) path is not implemented.')

    n_points, n_boxes = boxes.shape[0], boxes.shape[1]
    gt_boxes6 = boxes[..., :6]                                        # (n_points, n_boxes, 6)
    pred_boxes = bbox_preds.unsqueeze(1).expand(n_points, n_boxes, 6)

    pred_lf = TSPHead._bbox_to_loss(pred_boxes)   # corner form (x1,y1,z1,x2,y2,z2)
    gt_lf = TSPHead._bbox_to_loss(gt_boxes6)

    # C_reg: DIoU loss (기존 bbox_loss와 동일한 정의 재사용)
    c_reg = axis_aligned_diou_loss(
        pred_lf.reshape(-1, 6), gt_lf.reshape(-1, 6), reduction='none'
    ).reshape(n_points, n_boxes)

    # R_VD: 정규화 vertex distance
    vd = torch.norm(pred_lf[..., :3] - gt_lf[..., :3], dim=-1) \
       + torch.norm(pred_lf[..., 3:] - gt_lf[..., 3:], dim=-1)
    enclose_min = torch.minimum(pred_lf[..., :3], gt_lf[..., :3])
    enclose_max = torch.maximum(pred_lf[..., 3:], gt_lf[..., 3:])
    rho = torch.norm(enclose_max - enclose_min, dim=-1)
    r_vd = vd / (2 * rho + 1e-6)

    # gamma_c: center prior (예측 박스 중심 vs GT 중심)
    center_dist_sq = torch.sum(
        torch.pow(pred_boxes[..., :3] - gt_boxes6[..., :3], 2), dim=-1)
    gamma_c = 1 - torch.exp(-mu * center_dist_sq)

    cost = gamma_c * (c_reg + r_vd)

    if alpha > 0 and cls_preds is not None:
        # 선택 ablation(기본 off): overlapping distractor 대응용 grounding term.
        grounding_term = alpha * (-F.logsigmoid(cls_preds).squeeze(-1))
        cost = cost + grounding_term.unsqueeze(1)

    return cost


class TR3DAssigner:
    def __init__(self, top_pts_threshold, label2level):
        # top_pts_threshold: per box
        # label2level: list of len n_classes
        #     scannet: [0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0]
        #     sunrgbd: [1, 1, 1, 0, 0, 1, 0, 0, 1, 0]
        #       s3dis: [1, 0, 1, 1, 0]
        self.top_pts_threshold = top_pts_threshold
        self.label2level = label2level

    @torch.no_grad()
    def assign(self, points, gt_bboxes, gt_labels, img_meta,
               bbox_preds=None, use_spota=False, k=6, mu=1.0, alpha=0.0,
               cls_preds=None):
        # -> object id or -1 for each point
        # bbox_preds/use_spota 등은 SPOTA(메인 voxel 배정) 전용. 기본값(use_spota=False)에서는
        # 기존 center-distance 동작과 100% 동일 — completion 배정은 항상 이 기본 경로를 탄다.
        float_max = points[0].new_tensor(1e8)
        levels = torch.cat([points[i].new_tensor(i, dtype=torch.long).expand(len(points[i]))
                            for i in range(len(points))])
        points = torch.cat(points)
        n_points = len(points)
        n_boxes = len(gt_bboxes)

        if len(gt_labels) == 0:
            return gt_labels.new_full((n_points,), -1)

        boxes = torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
        boxes = boxes.to(points.device).expand(n_points, n_boxes, 7)
        points = points.unsqueeze(1).expand(n_points, n_boxes, 3)

        # condition 1: fix level for label
        label2level = gt_labels.new_tensor(self.label2level)
        label_levels = label2level[gt_labels].unsqueeze(0).expand(n_points, n_boxes)
        point_levels = torch.unsqueeze(levels, 1).expand(n_points, n_boxes)
        level_condition = label_levels == point_levels

        # primary: box를 고르는 기준값(작을수록 좋음). off=center-distance, SPOTA on=cost.
        if use_spota and bbox_preds is not None:
            primary_raw = _spota_cost(points, boxes, bbox_preds, mu, alpha, cls_preds)
            top_k = k
        else:
            center = boxes[..., :3]
            primary_raw = torch.sum(torch.pow(center - points, 2), dim=-1)
            top_k = self.top_pts_threshold

        # condition 2: keep topk location per box by primary (level 조건 적용)
        primary = torch.where(level_condition, primary_raw, float_max)
        topk_vals = torch.topk(primary,
                               min(top_k + 1, len(primary)),
                               largest=False, dim=0).values[-1]
        topk_condition = primary < topk_vals.unsqueeze(0)

        # condition 3.0: only closest object to point (level 조건 무시, raw 기준 — 기존과 동일)
        _, min_inds_ = primary_raw.min(dim=1)

        # condition 3: min primary to box per point, among topk-selected
        primary_topk = torch.where(topk_condition, primary_raw, float_max)
        min_values, min_ids = primary_topk.min(dim=1)
        min_inds = torch.where(min_values < float_max, min_ids, -1)
        min_inds = torch.where(min_inds == min_inds_, min_ids, -1)

        return min_inds
