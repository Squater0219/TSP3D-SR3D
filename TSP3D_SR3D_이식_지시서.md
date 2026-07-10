# TSP3D에 SR3D 훈련법(SPOTA + RAS) 이식 — Claude Code 작업 지시서

## 0. 목표 (한 줄)

TSP3D(3D Visual Grounding)의 **추론 구조와 속도는 그대로 두고**, 학습 단계의 **label assignment**와 **분류 손실**만 SR3D(AAAI 2026)의 방식으로 교체한다. 추론 시점 비용 증가는 0이어야 한다(SPOTA/RAS는 training-only).

교체 대상은 단 두 곳이다:
- **SPOTA**: 현재 center-distance top-k 배정 → spatial-priority cost 기반 top-k 배정
- **RAS**: 현재 voxel 분류 FocalLoss → rank-aware self-distillation 손실

---

## 1. 배경 — 무엇을, 왜

- TSP3D는 sparse-conv 기반 single-stage 3DVG 모델이다. text query로 지칭된 단일 객체의 3D 박스를 찾는다.
- SR3D는 동일 계열(TR3D 기반) 실내 3D **detection**에서, 모델을 안 바꾸고 학습 절차만 바꿔 정확도를 올렸다. 핵심 진단은 "학습과 평가(AP, ranking 기반)가 어긋나 있다"는 것이고, 두 가지로 해결한다.
  - **SPOTA** (Spatial-Prioritized OTA): 중심거리 같은 고정 휴리스틱 대신, 기하적 신뢰도(vertex distance + IoU + center prior)로 positive를 동적으로 고른다.
  - **RAS** (Rank-aware Adaptive Self-Distillation): 분류 confidence를 localization 정확도(IoU)의 순위에 정렬시킨다. 잘 못 맞춘 박스에 높은 confidence를 주는 것을 억제.
- 우리는 이 두 절차를 TSP3D(detection이 아니라 grounding)에 옮긴다. 3DVG는 query당 target이 1개라 **본질적으로 ranking 문제**이고, SR3D의 동기가 detection보다 더 직접적으로 들어맞는다.

---

## 2. 절대 원칙 (작업 전 반드시 숙지)

1. **모든 신규 기능은 플래그로 on/off 가능해야 한다.** 기본값은 전부 **off**. 플래그가 전부 off이면 **기존 baseline과 비트 단위로 동일하게** 동작해야 한다. (논문 ablation의 유효성이 여기 달려 있다.)
2. **추론 경로(`forward_test`, `_get_bboxes*`)는 절대 수정하지 않는다.** SPOTA/RAS는 학습에만 개입한다.
3. **기존 pruning(TGP), completion(CBA), text fusion, backbone은 건드리지 않는다.** 우리가 바꾸는 건 (a) assignment, (b) 분류 손실 두 가지뿐이다.
4. **데이터셋은 ScanRefer(axis-aligned, `with_yaw=False`, `box_dim=6`)를 기준으로 한다.** NR3D/SR3D의 rotated 경로를 깨뜨리지 말되, 새 코드는 axis-aligned를 가정해 단순하게 구현한다.
5. 기존 함수 시그니처를 바꿔야 하면, **기본값 인자를 추가**하는 방식으로 하위호환을 유지한다(기존 호출부가 안 깨지게).
6. 새 함수/유틸은 별도로 추가하고, 기존 코드 경로는 `if use_xxx:` 분기로 보존한다. 기존 라인을 통째로 덮어쓰지 말 것.

---

## 3. 사전 분석으로 확정된 코드 사실 (재조사 불필요, 신뢰해도 됨)

모든 핵심 로직은 `models/multilevel_head.py`에 있다. 아래는 코드를 직접 읽고 확인한 사실이다.

### 3-1. Head 구조: single-final head + multi-level pruning
- `forward`(약 line 218~390)에서 pruning은 level 2→1→0으로 단계적이지만, **박스/스코어 예측은 finest level(level 0)에서 단 한 번만** 일어난다.
- 끝부분(약 line 388~390):
  ```python
  out = self.fuse(out, text_feats[:, 0])
  bbox_pred, cls_pred, point = self._forward_single(out)
  return [bbox_pred], [cls_pred], [point], keep_preds[::-1], keep_gts[::-1], ...
  ```
- `[bbox_pred]`, `[cls_pred]`는 리스트지만 **원소 1개**다.
- **→ SPOTA는 "어느 level에 걸지" 고민이 필요 없다. final head의 예측 voxel 집합에 그대로 적용하면 된다.**

### 3-2. 현재 assignment: 순수 center-distance top-k (`TR3DAssigner`)
- 클래스 `TR3DAssigner`(약 line 909~961). 메서드 `assign`은 `@torch.no_grad()`이고 인자는 `(points, gt_bboxes, gt_labels, img_meta)`로, **예측 박스를 받지 않는다.**
- 동작:
  - condition 1: `label2level=[0]` → 모든 GT가 level 0. single-level이라 사실상 항상 참.
  - condition 2: box 중심 ↔ voxel **중심거리**로 `top_pts_threshold(=32)`개 선택. **IoU/shape 미사용.**
  - condition 3: 각 voxel을 가장 가까운 box 하나에만 배정.
- **→ 이 condition 2의 center-distance top-k를 SPOTA cost 기반 top-k로 교체하는 것이 핵심.**

### 3-3. Grounding score: voxel별 single logit (`cls_conv`, `n_classes=1`)
- `__init__`에서 `n_classes=1`. `_init_layers`(line 135): `self.cls_conv = ME.MinkowskiConvolution(out_channels, 1, ...)`.
- `_forward_single`(line 207~208): `scores = self.cls_conv(x); cls_pred = scores.features` → voxel당 1차원 logit.
- 즉 18-class가 아니라 **"이 voxel이 지칭된 target에 속하는가"의 binary score**.
- 추론(`_get_bboxes_single`, line 741~747): `scores = cls_preds.sigmoid()` 후 `topk(nms_pre=1)`로 **최고 score voxel 1개**를 target으로.
- **→ RAS의 분류 score는 이 `cls_conv` logit이다. SR3D의 class score와 구조가 1:1 대응 → RAS 거의 그대로 이식 가능.**

### 3-4. 손실 계산: `_loss_single`(line 568~618), `_loss`(line 621~665)
- `_loss_single`에서 **assign이 두 번 호출됨**에 주의:
  - line 576: `assigned_ids = self.assigner.assign(points, gt_bboxes, gt_labels, img_meta)` → **메인 voxel 배정** (이 voxel들은 박스 예측 `bbox_preds`를 가진다)
  - line 592: `assigned_ids_com = self.assigner.assign([com_coords], gt_bboxes, gt_labels, img_meta)` → **completion voxel 배정** (이 voxel들은 별도 박스 예측이 없고, completion score만 예측)
- 분류 손실: line 590 `cls_loss = self.cls_loss(cls_preds, cls_targets)` (FocalLoss). pos voxel은 target=0(foreground), neg voxel은 target=`n_classes`(=1, background).
- 박스 손실: line 613 `self.bbox_loss = AxisAlignedIoULoss2(mode='diou', ...)` 사용. (`_loss_single`에서 `bbox_loss` 계산.)
- **→ SPOTA는 박스 예측이 필요하므로 메인 배정(line 576)에만 적용한다. completion 배정(line 592)은 박스 예측이 없으니 현재 center-distance 방식을 유지한다.**

### 3-5. Box 표현: axis-aligned + DIoU loss가 이미 존재
- ScanRefer 경로는 `box_dim=6, with_yaw=False`(line 728~729, 754).
- `self.bbox_loss = AxisAlignedIoULoss2(mode='diou', reduction='none')`(line 86). `axis_aligned_diou_loss`는 `models/axis_aligned_iou_loss.py`(약 line 177)에 정의돼 있음.
- IoU 계산은 같은 파일의 `axis_aligned_bbox_overlaps_3d`(약 line 131~149) 사용 가능.
- 예측 파라미터 → 실제 박스 변환은 `_bbox_pred_to_bbox`(static, line 529)와 `_bbox_to_loss`(static, line 509) 사용.
- **→ SPOTA의 `C_reg`와 RAS의 `q(IoU)`에 필요한 인프라가 이미 다 있다. 새로 구현할 건 vertex distance와 soft rank뿐.**

---

## 4. 작업 A — `soft_rank` 유틸 추가

RAS와 (선택적으로) SPOTA에서 공통으로 쓰는 soft ranking 함수. SR3D 보충자료 Eq.9~10 그대로.

`models/multilevel_head.py` 상단(클래스 밖) 또는 `TSPHead`의 static method로 추가:

```python
def soft_rank(scores, tau=0.1, eps=1e-6):
    """
    내림차순 soft rank. SR3D 보충자료 Eq.9-10.
    scores: 1D tensor (점수가 클수록 '좋음', rank가 높음)
    반환 r: 1D tensor, r in (0,1], 점수가 클수록 r이 1에 가까움.
      R_i = (1/N) * sum_{j != i} sigmoid((s_j - s_i)/tau)   # 나보다 큰 점수의 비율
      r_i = exp(-R_i)
    tau -> 0 이면 hard rank에 수렴. 작은 점수일수록 R_i가 커져 r_i가 작아진다.
    """
    if scores.numel() == 0:
        return scores
    s = scores.view(-1)
    N = s.numel()
    if N == 1:
        return torch.ones_like(s)
    diff = s.unsqueeze(0) - s.unsqueeze(1)          # diff[j, i] = s_j - s_i
    R = torch.sigmoid(diff / tau).sum(dim=0)        # sum over j
    R = (R - torch.sigmoid(torch.zeros(1, device=s.device))) / max(N - 1, 1)  # j != i 보정
    r = torch.exp(-R)
    return r
```

주의: `j != i` 항만 합산해야 하므로 대각 성분(`sigmoid(0)=0.5`)을 빼고 `N-1`로 정규화한다. 위 구현은 그 보정을 포함한다. 구현이 헷갈리면 명시적으로 대각 마스크를 써도 된다:
```python
mask = ~torch.eye(N, dtype=torch.bool, device=s.device)
R = (torch.sigmoid(diff / tau) * mask).sum(dim=0) / (N - 1)
```

---

## 5. 작업 B — RAS (rank-aware 분류 손실) [먼저 구현]

> RAS가 SPOTA보다 쉽다. **구조 변경 없이 `_loss_single`의 분류 손실만 교체**하면 된다. 이걸 먼저 완성하고 검증한다.

### 5-1. 무엇을 하는가
positive voxel에 대해, 기존 FocalLoss를 **focal과 rank-aware self-distillation의 적응 혼합**으로 바꾼다. negative voxel은 FocalLoss 유지.

### 5-2. 필요한 값
positive voxel 집합 `P`에 대해:
- `sig_i = sigmoid(cls_logit_i)` : 분류 confidence (스칼라)
- `q_i = IoU(예측박스_i, 배정된 GT박스)` : localization quality. `_bbox_pred_to_bbox`로 예측박스 만들고 `axis_aligned_bbox_overlaps_3d(..., is_aligned=True)`로 IoU 계산
- `r_reg_i = soft_rank(q over P, tau)` : IoU의 soft rank (잘 맞출수록 큼)
- `r_cls_i = soft_rank(sig over P, tau)` : confidence의 soft rank

### 5-3. 손실 수식 (SR3D Eq.6, Eq.7)
self-distillation 손실(positive only):
```
RDL_i = (1 - r_reg_i)^beta * [ q_i * log(sig_i) ] + q_i * (1 - q_i) * log(1 - sig_i)
```
(논문 표기 그대로. 부호는 cross-entropy 형태이므로 최종적으로 `- RDL_i`를 최소화하도록 구현. 즉 손실 = `-(위 식)`. focal loss와 부호 규약을 맞출 것.)

최종 분류 손실:
```
L_cls = sum_{i in P} [ (1 - r_cls_i) * FL_i + r_cls_i * RDL_loss_i ]  +  sum_{j in N} FL_j
```
- `FL_i`: 기존 FocalLoss를 positive voxel에 대해 계산한 값
- `RDL_loss_i`: 위 RDL을 손실 형태(`-RDL_i`)로 만든 값
- negative는 기존 FocalLoss 그대로

### 5-4. 구현 위치와 방법
- `_loss_single`(line 568~618)에서 **메인 분류 손실**(line 590 부근)만 분기한다.
  ```python
  if self.use_ras and pos_mask.sum() > 0:
      # 1) 예측 박스 만들고 q(IoU) 계산
      # 2) r_reg, r_cls = soft_rank(...)
      # 3) RDL, FL 혼합으로 positive 손실 계산
      # 4) negative는 기존 FocalLoss
      # 5) 합쳐서 cls_loss 구성 (기존 reduction 규약과 동일하게 cat 형태로 반환)
  else:
      cls_loss = self.cls_loss(cls_preds, cls_targets)   # 기존 경로 그대로
  ```
- **completion 손실(`com_loss`, line 601)은 일단 그대로 둔다.** (선택적으로 나중에 RAS를 com에도 적용하는 ablation을 고려할 수 있으나 1차 범위 밖.)
- `_loss`(line 661~665)의 reduction 방식(`torch.sum(cat) / torch.sum(pos_masks)`)과 호환되도록, `_loss_single`이 반환하는 `cls_loss` 텐서의 형태/스케일을 기존과 맞춘다.

### 5-5. RAS가 3DVG에서 주는 보너스 (구현엔 영향 없지만 인지할 것)
RAS는 grounding score를 "target과의 IoU" 순위에 정렬시킨다. distractor(같은 클래스 다른 객체)는 target과 IoU가 낮으니 자연히 낮은 score를 받게 되어, **ScanRefer의 "multiple" subset(distractor 존재) 개선이 기대된다.** ablation에서 unique/multiple subset을 나눠 보고하면 좋다.

---

## 6. 작업 C — SPOTA (spatial-priority 배정) [RAS 검증 후 구현]

> SPOTA는 assigner가 **예측 박스**를 봐야 하므로 dataflow 변경이 필요하다. RAS가 안정적으로 돌아간 뒤에 착수한다.

### 6-1. 무엇을 하는가
`TR3DAssigner.assign`의 condition 2(center-distance top-k)를 **SPOTA cost 기반 top-k**로 교체한다.

### 6-2. 시그니처 변경 (하위호환 유지)
```python
@torch.no_grad()
def assign(self, points, gt_bboxes, gt_labels, img_meta,
           bbox_preds=None, use_spota=False, k=6, mu=1.0):
    ...
```
- `bbox_preds=None`이거나 `use_spota=False`이면 **현재 center-distance 동작 그대로** (completion 배정·baseline 보존).
- `use_spota=True` 그리고 `bbox_preds`가 주어지면 SPOTA cost로 top-k 선택.

### 6-3. SPOTA cost 수식 (SR3D 본문 Eq.3, Eq.4, Eq.5)
axis-aligned 박스를 두 대각 corner로 표현:
```
v1 = (x_min, y_min, z_min),  v2 = (x_max, y_max, z_max)
```
정규화 vertex distance:
```
R_VD = ( ||v̂1 - v1*|| + ||v̂2 - v2*|| ) / ( 2 * rho(b̂, b*) )
```
- `v̂*`: 예측 박스 corner, `v*`: GT 박스 corner
- `rho(b̂, b*)`: 두 박스를 모두 감싸는 **최소 외접 박스의 대각선 길이** (즉 두 박스 합집합 bounding box의 corner-to-corner 거리)

center prior:
```
gamma_c = 1 - exp( -mu * ||c - c_gt||^2 )
```
- `c`: 예측 박스(또는 anchor voxel) 중심, `c_gt`: GT 박스 중심

최종 cost:
```
C = gamma_c * ( C_reg + R_VD )
```
- `C_reg`: DIoU loss(예측박스 vs GT). `axis_aligned_diou_loss` 재사용.
- per-GT로 **C가 가장 작은 top-k voxel을 positive**로 선정 (`k=6` 기본).
- **분류(text) cost는 넣지 않는다.** (SR3D 주장: 3D는 geometry가 semantic을 인코딩. 우리 head도 이미 text-fused feature 위에서 박스를 예측하므로 cost가 text-blind가 아니다.)

> 선택 ablation(기본 off): overlapping distractor 대응용으로 작은 grounding term `C += alpha * (-log sigmoid(s_i))`를 더해볼 수 있게 `alpha` 인자(기본 0.0)만 만들어 둔다. 기본 경로는 `alpha=0`.

### 6-4. 학습 초기 불안정 처리 (중요)
학습 초기엔 예측 박스가 엉망이라 R_VD/IoU가 신뢰 불가다. SR3D는 `gamma_c`(center prior)로 이를 완화한다(μ가 클수록 center 의존↑). 즉 **초기엔 사실상 center 기반 → 학습되며 vertex/IoU 기반으로 자연 전환**된다. 별도 warmup 스케줄을 짜지 말고, cost 식에 `gamma_c`를 곱하는 것으로 충분하다(논문도 그렇게 함). 기본 `mu=1.0`.

### 6-5. 호출부 수정
- `_loss_single`(line 576)의 메인 assign 호출에 예측 박스와 플래그를 전달:
  ```python
  if self.use_spota:
      pred_boxes = self._bbox_pred_to_bbox(torch.cat(points), torch.cat(bbox_preds))
      assigned_ids = self.assigner.assign(
          points, gt_bboxes, gt_labels, img_meta,
          bbox_preds=pred_boxes, use_spota=True, k=self.spota_k, mu=self.spota_mu)
  else:
      assigned_ids = self.assigner.assign(points, gt_bboxes, gt_labels, img_meta)
  ```
  (예측 박스를 밖에서 만들어 넘기거나, `bbox_preds`(원시 예측)와 `points`를 넘겨 assign 내부에서 `_bbox_pred_to_bbox`로 변환하거나 둘 중 하나. 일관되게.)
- **completion assign(line 592)은 수정하지 않는다** → `bbox_preds=None`으로 기존 center-distance 유지.
- `assign`은 `@torch.no_grad()`이므로 예측 박스를 `detach`해서 넘기는 것과 동치다(그래디언트 안 흐름). 그대로 둬도 되지만, 명시적으로 detach해도 무방.

### 6-6. SPOTA가 pruning과 충돌하지 않는가
- SPOTA는 final head의 **생존 voxel**(TGP/CBA 통과분)에 대해서만 동작한다. CBA가 over-pruning된 target 영역을 복원하므로 target 주변 voxel은 대체로 살아 있다.
- 1차 구현은 **pruning과 decouple**한다: 기존 `keep_loss`/`com_loss`는 그대로 두고, SPOTA는 분류/박스 손실의 positive 선정에만 관여한다.
- (선택, stretch) pruning supervision을 SPOTA positive와 일관되게 만드는 coupled 변형은 `_get_keep_voxel`(line 449)을 손대야 하므로 **1차 범위 밖**. 별도 플래그 `use_spota_coupled`(기본 off) 자리만 남겨두고 구현은 보류.

---

## 7. 작업 D — Ablation 플래그 배선

### 7-1. `TSPHead.__init__`에 인자 추가 (기본 전부 off/논문값)
```python
def __init__(self, ...,
             use_spota=False, use_ras=False,
             spota_k=6, spota_mu=1.0, spota_alpha=0.0,
             ras_beta=1.0, ras_tau=0.1,
             ...):
    ...
    self.use_spota = use_spota
    self.use_ras = use_ras
    self.spota_k = spota_k
    self.spota_mu = spota_mu
    self.spota_alpha = spota_alpha
    self.ras_beta = ras_beta
    self.ras_tau = ras_tau
```
- 하이퍼파라미터 기본값은 SR3D 논문 권장: `k=6, mu=1.0, beta=1.0, tau=0.1`. (SR3D는 최종 default로 `tau=0.01`도 언급하니 인자로 노출만 해두면 됨.)

### 7-2. 상위로 배선
- `models/bdetr.py`의 `BeaUTyDETR.__init__`(약 line 20~54)에 동일 인자를 추가하고 `TSPHead(...)` 생성(line 54)에 전달.
- `BeaUTyDETR`를 생성하는 지점(`train_dist_mod.py` 또는 `models/__init__.py`)까지 인자를 전달하고, **학습 스크립트의 argparse/명령행 옵션**으로 노출한다. 예: `--use_ras`, `--use_spota`, `--spota_k`, `--ras_beta`, `--ras_tau` 등.
- 어디서 모델이 생성되는지 코드를 따라가서 배선할 것. (`train_dist_mod.py`와 `models/__init__.py`를 확인하라.)

### 7-3. 검증 가능한 조합
- `--use_ras`만
- `--use_spota`만
- 둘 다 (full)
- 둘 다 off → **baseline과 동일** (반드시 확인)

---

## 8. 작업 E — Case Study 실험 (SR3D Table 6 재현, upper bound 측정)

SR3D는 "박스는 잘 그리는데 confidence를 못 매긴다"를 증명하려고, 추론 직전 분류 score를 GT IoU로 치환해 성능 상한을 봤다(detection에서 70.8→91.8 AP25).

우리 버전: **추론 시 `cls_pred`(grounding score)를 "예측박스와 GT의 IoU"로 치환**한 뒤 평가하는 디버그 모드를 추가한다.
- `forward_test`나 평가 루프에 `--oracle_score` 플래그(기본 off)를 만들어, 켜면 각 예측 voxel의 score를 GT IoU로 대체.
- GT 박스는 평가 시점에 접근 가능해야 하므로(eval 전용), `src/grounding_evaluator.py` 또는 평가 스크립트에서 처리하는 게 깔끔하다. **이 모드는 디버그/분석 전용이며 학습/정식 추론과 분리**한다.
- 목적: 3DVG에서도 grounding score calibration이 주요 병목인지 정량 확인. 이 수치 하나가 논문 motivation의 강력한 근거가 된다.

> 이 작업은 모델 변경이 아니라 분석 실험이다. 우선순위는 RAS/SPOTA 다음.

---

## 9. 함정 / 주의사항

1. **`assign`은 `@torch.no_grad()`다.** SPOTA에서 예측 박스를 넣어도 그래디언트가 안 흐른다(의도된 동작, SR3D도 동일). assignment는 "어느 voxel이 positive인가"만 정하고, 실제 학습 신호는 그 뒤 손실에서 나온다.
2. **assign이 두 번 불린다.** 메인(line 576, 박스 예측 있음)과 completion(line 592, 박스 예측 없음). SPOTA는 메인에만. completion은 `bbox_preds=None`으로 기존 동작 유지.
3. **`n_classes=1`이라 pos target=0, neg target=1(=n_classes).** RAS 구현 시 positive/negative 마스크와 target 인덱스 규약을 기존 FocalLoss 호출과 동일하게 유지.
4. **`_forward_single`에 `reg_angle = reg_final[:, 6:]` 코드가 있지만 `n_reg_outs=6`이라 빈 텐서다(rotated 경로용 dead code).** ScanRefer(axis-aligned)에서는 무시. 새 코드는 6-dim 박스를 가정.
5. **reduction 스케일을 깨지 마라.** `_loss`(line 661~665)는 `cls_loss = sum(cat) / sum(pos_masks)`로 정규화한다. RAS로 분류 손실 형태를 바꿀 때 이 정규화와 호환되게 텐서를 반환할 것.
6. **`rho`(최소 외접 박스 대각선) 계산 시 0 division 주의.** 두 박스가 거의 일치하면 분모가 작아질 수 있으니 `eps`를 더한다.
7. **베이스라인 동등성은 코드 리뷰가 아니라 수치로 확인하라.** 플래그 off에서 동일 seed로 몇 step 돌려 loss 값이 기존과 일치하는지 본다.
8. **NR3D/SR3D 스크립트를 깨지 마라.** rotated 경로(box_dim=7)가 존재한다. 새 코드는 axis-aligned 가정이지만, `with_yaw=True` 경로에서 NotImplemented로 떨어지더라도 최소한 기존 baseline 경로는 보존돼야 한다(SPOTA/RAS off일 때).

---

## 10. 검증 체크리스트 (작업 완료 기준)

- [ ] `soft_rank` 단위 테스트: 단조 입력에 대해 큰 값일수록 r이 1에 가까운지, `N=1` 엣지케이스 통과.
- [ ] **모든 플래그 off → baseline과 loss/지표 동일** (동일 seed, 몇 step).
- [ ] `--use_ras`만 켜고 학습이 NaN 없이 진행, ScanRefer val Acc@0.25/0.5 측정.
- [ ] `--use_spota`만 켜고 학습 안정성 확인(초기 불안정이 `gamma_c`로 완화되는지). val 지표 측정.
- [ ] full(둘 다) 학습 후 baseline 대비 지표 비교.
- [ ] unique vs multiple subset 분리 보고 (RAS의 distractor 억제 효과 확인).
- [ ] `--oracle_score`로 upper bound 측정(Case Study).
- [ ] 하이퍼파라미터 노출 확인: `k, mu, beta, tau`를 명령행에서 바꿀 수 있는지.
- [ ] 추론 latency가 baseline과 동일한지(SPOTA/RAS는 training-only이므로 변화 0이어야 함).

---

## 부록 — 수식 모음 (빠른 참조)

**soft rank (내림차순, SR3D Eq.9-10)**
```
R_i = (1/(N-1)) * sum_{j != i} sigmoid((s_j - s_i)/tau)
r_i = exp(-R_i)
```

**SPOTA cost (SR3D Eq.3-5)**
```
R_VD    = ( ||v̂1 - v1*|| + ||v̂2 - v2*|| ) / ( 2 * rho(b̂, b*) )
gamma_c = 1 - exp( -mu * ||c - c_gt||^2 )
C       = gamma_c * ( C_reg + R_VD )           # C_reg = DIoU loss
positive = per-GT로 C 최소 top-k (k=6)
```

**RAS 손실 (SR3D Eq.6-7)**
```
RDL_i  = (1 - r_reg_i)^beta * q_i*log(sig_i) + q_i*(1 - q_i)*log(1 - sig_i)
L_cls  = sum_{i in P} [ (1 - r_cls_i)*FL_i + r_cls_i*(-RDL_i) ] + sum_{j in N} FL_j
         # q = IoU(예측, GT),  sig = sigmoid(cls_logit),  r_* = soft_rank
```

**기본 하이퍼파라미터**: `k=6, mu=1.0, beta=1.0, tau=0.1`

---

## 작업 순서 요약 (이대로 진행)

1. **작업 A** (`soft_rank`) → 단위 테스트
2. **작업 D 일부** (RAS 플래그만 먼저 배선)
3. **작업 B** (RAS) → 베이스라인 동등성 확인 → `--use_ras` 학습/평가
4. **작업 D 나머지** (SPOTA 플래그 배선)
5. **작업 C** (SPOTA) → 학습 안정성 확인 → `--use_spota` 학습/평가
6. full 학습/평가, unique/multiple 분리
7. **작업 E** (Case Study, oracle score)
