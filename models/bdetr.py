import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizerFast
import MinkowskiEngine as ME
from .mink_resnet import TSPBackbone
from .tr3d_neck import TR3DNeck
from .multilevel_head import TSPHead
from mmdet3d.structures.bbox_3d import DepthInstance3DBoxes
from mmdet3d.structures import bbox3d2result
import time
import pdb
    
class BeaUTyDETR(nn.Module):
    """
    3D language grounder.
    """

    def __init__(self, num_class=256, num_obj_class=485,
                 input_feature_dim=3,
                 num_queries=256,
                 num_decoder_layers=6, self_position_embedding='loc_learned',
                 contrastive_align_loss=True,
                 d_model=128, butd=True, pointnet_ckpt=None, data_path=None,
                 self_attend=True, voxel_size=0.01,
                 use_spota=False, use_ras=False,
                 spota_k=6, spota_mu=1.0, spota_alpha=0.0,
                 spota_greedy_topk=False,
                 ras_beta=1.0, ras_tau=0.1):
        """Initialize layers."""
        super().__init__()

        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers
        self.self_position_embedding = self_position_embedding
        self.contrastive_align_loss = contrastive_align_loss
        self.butd = butd
        self.voxel_size = voxel_size

        # Visual encoder
        self.vision_backbone = TSPBackbone(in_channels=6)
        
        # Text encoder
        t_type = f'{data_path}roberta-base/'
        self.tokenizer = RobertaTokenizerFast.from_pretrained(t_type, local_files_only=True)
        self.text_encoder = RobertaModel.from_pretrained(t_type, local_files_only=True)
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.text_projector = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, d_model),
            nn.LayerNorm(d_model, eps=1e-12),
            nn.Dropout(0.1)
        )       
        
        # self.neck = TR3DNeck()
        self.head = TSPHead(
            voxel_size=self.voxel_size,
            use_spota=use_spota, use_ras=use_ras,
            spota_k=spota_k, spota_mu=spota_mu, spota_alpha=spota_alpha,
            spota_greedy_topk=spota_greedy_topk,
            ras_beta=ras_beta, ras_tau=ras_tau)
        
    
    # BRIEF forward.
    def forward(self, inputs, gt_bboxes=None, gt_labels=None, gt_all_bbox_new=None, auxi_bbox=None, img_metas=None, epoch=None):
        """
        Forward pass.
        Args:
            inputs: dict
                {point_clouds, text}
                point_clouds (tensor): (B, Npoint, 3 + input_channels)
                text (list): ['text0', 'text1', ...], len(text) = B
        Returns:
            end_points: dict
        """
        points = inputs['point_clouds']
        start_time = time.time()

        #Step 4: 포인트 클라우드 복셀화 — Dense 포인트를 Sparse Voxel로 양자화
        # points: (B,N,6) xyz+RGB → ME.SparseTensor: (N_voxels, 6), voxel_size=0.01m 격자로 양자화
        # 동일 복셀에 속하는 포인트는 ME 내부에서 합산. coordinates 첫 열에 배치 인덱스 포함.
        coordinates, features = ME.utils.batch_sparse_collate(
                [(p[:, :3] / self.voxel_size, p[:, 0:] if p.shape[1] > 3 else p[:, :3]) for p in points],
                device=points[0].device)
        x = ME.SparseTensor(coordinates=coordinates, features=features)

        #Step 5: Sparse 3D 백본 추론 — TSPBackbone(MinkowskiEngine ResNet34)으로 4레벨 다중 스케일 특징 추출
        # ME.SparseTensor(N_voxels, 6) → list[4] SparseTensor
        #
        # 레이어 구조 (depth=34, max_channels=128, voxel_size=0.01m):
        #   Stem  : conv1(k=3,s=2,ch=64) → BN → ReLU → MaxPool(k=2,s=2)
        #           N_voxels 유지, tensor_stride=4, 실제 격자 크기=0.04m
        #   layer1: BasicBlock×3, ch=64,  stride=2 → tensor_stride=8,  격자=0.08m  → outs[0]
        #   layer2: BasicBlock×4, ch=128, stride=2 → tensor_stride=16, 격자=0.16m  → outs[1]
        #   layer3: BasicBlock×6, ch=128, stride=2 → tensor_stride=32, 격자=0.32m  → outs[2]
        #   layer4: BasicBlock×3, ch=128, stride=2 → tensor_stride=64, 격자=0.64m  → outs[3]
        #
        # TSPHead에서는 outs[1:] (layer2~4)만 사용:
        #   inputs[0]=outs[1](128ch,0.16m) ← 레벨0(가장 세밀, Completion branch 원본 소스)
        #   inputs[1]=outs[2](128ch,0.32m) ← 레벨1
        #   inputs[2]=outs[3](128ch,0.64m) ← 레벨2(가장 거침, 루프 시작점)
        x = self.vision_backbone(x)
        visual_time = time.time() - start_time

        start_time = time.time()

        #Step 6: 텍스트 토크나이징 — 자연어 표현을 RoBERTa 토큰 시퀀스로 변환
        # text: list[B] str → input_ids: (B,L) int, attention_mask: (B,L) int
        # padding="longest": 배치 내 최장 문장 길이 L에 맞춰 짧은 문장을 패딩.
        tokenized = self.tokenizer.batch_encode_plus(
            inputs['text'], padding="longest", return_tensors="pt"
        ).to(inputs['point_clouds'].device)

        #Step 7: RoBERTa 텍스트 인코딩 — 토큰을 문맥적 언어 임베딩으로 변환
        # input_ids: (B,L) → last_hidden_state: (B,L,768)
        # text_encoder는 frozen(requires_grad=False). 사전학습된 의미론적 표현을 고정 사용.
        encoded_text = self.text_encoder(**tokenized)

        #Step 8: 텍스트 특징 Projection — 768차원 → 비주얼 특징과 동일한 128차원으로 압축
        # last_hidden_state: (B,L,768) → text_feats: (B,L,128), text_attention_mask: (B,L) bool
        # Linear→LayerNorm→Dropout. attention_mask.ne(1): True=패딩 위치 (cross-attention 시 무시됨).
        text_feats = self.text_projector(encoded_text.last_hidden_state)
        text_attention_mask = tokenized.attention_mask.ne(1).bool()
        text_time = time.time() - start_time

        if not self.training:
            start_time = time.time()
            bbox_list, head_time = self.head.forward_test(x, text_feats, text_attention_mask, img_metas)
            bbox_results = [
                bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels in bbox_list
            ]
            fusion_time = time.time() - start_time
            return bbox_results, {'loss':0.}, 0., [visual_time,text_time,fusion_time-head_time,head_time]

        #Step 9~25: TSPHead Forward Train — 언어-유도 다중 스케일 프루닝 + 박스 예측 + 손실 계산
        # x: list[4] SparseTensor, text_feats: (B,L,128) → losses: {bbox_loss, cls_loss, keep_loss, com_loss}
        # 내부 순서: GT 레벨 분류(Step 9) → 레벨2→1→0 BiEncoder·프루닝(Step 10~22) → 박스 예측(Step 23~25) → 손실(Step 26~28)
        losses = self.head.forward_train(x,text_feats, text_attention_mask, gt_bboxes, gt_labels, gt_all_bbox_new, auxi_bbox, img_metas)
        losses.update({'loss':sum(value for key, value in losses.items() if '_loss' in key)})
        return losses
    def init_bn_momentum(self):
        """Initialize batch-norm momentum."""
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = 0.1
