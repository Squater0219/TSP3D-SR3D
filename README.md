# TSP3D + SR3D (SPOTA / RAS) — 이 fork에 대해

이 저장소는 [TSP3D](https://github.com/GWxuan/TSP3D)(CVPR 2025 Highlight)를 기반으로,
[SR3D](https://arxiv.org/abs/2502.10392)(AAAI 2026, 실내 3D detection)에서 제안한 학습 절차 두 가지 —
**SPOTA**(label assignment)와 **RAS**(분류 손실) — 를 3D visual grounding(TSP3D)에 이식한 fork입니다.

원본 TSP3D의 README는 [README_TSP.md](README_TSP.md)에 그대로 유지되어 있습니다(설치/데이터 준비/원본 학습·평가 방법은 그쪽 참고).
이 문서는 **이 fork에서 추가된 부분만** 다룹니다.

## 무엇을, 왜

TSP3D는 sparse-conv 기반 single-stage 3D visual grounding 모델로, 추론 구조와 속도는 그대로 두고 싶습니다.
SR3D의 핵심 진단은 "학습과 평가(ranking 기반 AP)가 어긋나 있다"는 것이고, 이를 두 가지로 해결합니다.

- **SPOTA** (Spatial-Prioritized OTA): 현재 TSP3D의 label assignment는 voxel-GT 박스 **중심거리** 기반 top-k 휴리스틱입니다.
  이를 예측 박스의 기하적 신뢰도(vertex distance + DIoU + center prior)로 동적으로 top-k를 고르는 방식으로 교체합니다.
- **RAS** (Rank-aware Adaptive Self-Distillation): 분류 confidence가 localization 정확도(IoU) 순위와 정렬되도록,
  기존 FocalLoss를 IoU 기반 self-distillation 손실과 적응적으로 혼합합니다.

3DVG는 query당 target이 1개라 본질적으로 ranking 문제이고, SR3D의 동기가 detection보다 더 직접적으로 들어맞습니다.

이식 작업의 상세 스펙/코드 위치/수식은 [TSP3D_SR3D_이식_지시서.md](TSP3D_SR3D_이식_지시서.md)에 있습니다.

## 절대 원칙

- **모든 신규 기능은 플래그로 on/off**, 기본값은 전부 **off**.
- 플래그가 전부 off이면 **기존 baseline과 동일하게** 동작합니다 (`TR3DAssigner.assign()`은 off일 때 리팩터링 전과
  bit-identical하도록 구현·검증했습니다).
- 추론 경로(`forward_test`, `_get_bboxes*`)와 기존 pruning(TGP)/completion(CBA)/text fusion/backbone은 건드리지 않았습니다.
  SPOTA/RAS는 **학습에만** 개입하는 training-only 기법입니다.

## 추가된 CLI 플래그

| 플래그 | 기본값 | 설명 |
|---|---|---|
| `--use_spota` | off | SPOTA(cost 기반 top-k) assignment 사용 |
| `--use_ras` | off | RAS(rank-aware self-distillation) 분류 손실 사용 |
| `--spota_k` | 6 | SPOTA top-k 개수 |
| `--spota_mu` | 1.0 | SPOTA center-prior 가중치(μ) |
| `--spota_alpha` | 0.0 | SPOTA 선택적 grounding term 가중치(기본 off) |
| `--ras_beta` | 1.0 | RAS self-distillation 항의 rank 지수(β) |
| `--ras_tau` | 0.1 | soft-rank 온도(τ), 작을수록 hard rank에 근접 |

## 사용 예시

```bash
# baseline (기존과 동일)
sh scripts/train_scanrefer_single.sh

# RAS만
python train_dist_mod.py ... --use_ras --ras_beta 1.0 --ras_tau 0.1

# SPOTA만
python train_dist_mod.py ... --use_spota --spota_k 6 --spota_mu 1.0

# full (SPOTA + RAS)
python train_dist_mod.py ... --use_spota --use_ras
```

## 어디를 고쳤는지

- `models/multilevel_head.py`
  - `soft_rank()`: 미분 가능한 내림차순 soft-rank 유틸 (SR3D 보충자료 Eq.9-10)
  - `TSPHead._ras_cls_loss()`: RAS 분류 손실 (Eq.6-7). positive voxel만 대상, completion(`com_loss`)은 미변경.
  - `_spota_cost()` + `TR3DAssigner.assign()`: SPOTA cost 계산 (Eq.3-5) 및 `bbox_preds/use_spota/k/mu/alpha` 인자 추가
    (기본값 하위호환). completion 배정은 항상 기존 center-distance 경로.
- `models/bdetr.py`, `train_dist_mod.py`, `main_utils.py`: 플래그를 argparse → `BeaUTyDETR` → `TSPHead`까지 배선.

## 검증 상태

구현 단계에서 확인한 것 (synthetic 텐서 기반 no-crash/단위 테스트):
- [x] `soft_rank` 단조성 단위 테스트, `N=1`/`N=0` 엣지케이스
- [x] `TSPHead` 생성자 off/on 정상
- [x] `assign()` 기본 호출 vs `use_spota=False` 명시 호출이 완전히 동일 (`torch.equal`)
- [x] SPOTA cost 경로(`alpha=0`/`alpha>0`) 크래시 없이 동작, rotated(7-dim) 입력엔 `NotImplementedError`
- [x] RAS 손실 forward/backward 정상, NaN/Inf 없음, positive 0개 엣지케이스 처리

아직 진행되지 않은 것 (수동 진행 예정):
- [ ] 플래그 전부 off일 때 실제 학습 스텝에서 baseline과 loss 수치 동일 확인
- [ ] `--use_ras`, `--use_spota`, full 학습 및 ScanRefer val Acc@0.25/0.5 평가
- [ ] unique/multiple subset 분리 결과
- [ ] oracle score(`--oracle_score`) case study — 아직 구현 안 됨 (지시서 작업 E, 우선순위 낮음)
