# FNO vs DeepONet: 1D PDE Operator Learning Benchmark

## 개요

이 저장소는 1차원 편미분방정식(PDE)에 대해 **Fourier Neural Operator(FNO)** 와 **Deep Operator Network(DeepONet)** 를 비교하기 위한 작은 규모의 실험 코드입니다.

실험의 핵심 관심사는 다음 두 가지입니다.

1. 입력 함수가 solution operator에 어떤 역할로 관여하는가?
2. PDE 연산자가 **공간 이동에 대해 상대적인 구조(translation-relative / shift-equivariant structure)** 를 가지는가?

특히 FNO가 Fourier 기반 구조를 사용하기 때문에, 위치 상대성이 있는 문제에서 강한 inductive bias를 보이는지, 반대로 절대 위치나 경계 조건이 중요한 문제에서는 coordinate channel이 도움이 되는지를 확인합니다.

---

## 실험 설계

총 4개의 1D PDE를 사용합니다. 각 PDE는 초기함수 의존성 여부와 위치 상대성 여부를 기준으로 선택했습니다.

| Config | PDE | 입력 함수의 역할 | 위치 상대성 | 설명 |
|---|---|---|---|---|
| `periodic_heat` | Periodic Heat Equation | 초기조건 | O | 주기 경계 조건을 가진 열 방정식 |
| `dirichlet_heat` | Dirichlet Heat Equation | 초기조건 | X | `u(0)=u(1)=0` 경계 조건을 가진 열 방정식 |
| `periodic_poisson` | Periodic Poisson Equation | forcing/source | O | 주기 경계 조건을 가진 Poisson 방정식 |
| `variable_poisson` | Variable-Coefficient Poisson Equation | coefficient/operator-changing input | X | 위치 의존 계수 `a(x)`를 가진 Poisson 방정식 |

---

## Coordinate Channel 실험

FNO는 Fourier/spectral convolution을 사용하기 때문에, 기본적으로 공간 이동에 대해 상대적인 구조와 잘 맞습니다.

하지만 Dirichlet boundary나 variable coefficient처럼 절대 위치 `x`가 중요한 문제에서는 순수 FNO 입력만으로는 위치 정보를 충분히 표현하기 어려울 수 있습니다.

이를 확인하기 위해 FNO에 대해 두 조건을 비교합니다.

```text
FNO without coordinate:
X = u(x)

FNO with coordinate:
X = [u(x), x]
````

즉, 입력 함수 값에 절대 좌표 `x`를 channel로 추가했을 때 성능이 개선되는지 확인합니다.

DeepONet은 trunk network가 좌표를 입력으로 받는 구조이므로, 이 실험에서는 주로 baseline operator learning model로 사용합니다.

---

## 디렉토리 구조

```text
.
├── configs/                  # 각 PDE 실험 설정 TOML 파일
├── data/raw/                 # 생성된 dataset artifact
├── checkpoints/              # 학습된 model checkpoint
├── results/
│   ├── logs/                 # 실행 로그, gitignore 권장
│   └── tables/               # 실험 요약 결과(summary.csv)
├── scripts/                  # 데이터 생성, 학습, 평가 실행 스크립트
├── src/
│   ├── models/               # FNO, DeepONet 구현
│   ├── pdes/                 # PDE solver 구현
│   ├── Config.jl
│   ├── DataGenerators.jl
│   ├── DatasetIO.jl
│   ├── Evaluate.jl
│   ├── Metrics.jl
│   ├── RandomFields.jl
│   └── Train.jl
├── test/                     # 간단한 테스트 코드
├── Project.toml
├── Manifest.toml
├── README.md
└── run_all_experiments.sh    # 전체 실험 자동 실행 스크립트
```

`data/raw/`, `checkpoints/`, `results/logs/`는 실행 과정에서 생성됩니다.

---

## 사용 방법

### 1. 환경 설정

Julia가 설치되어 있어야 합니다.

프로젝트 루트에서 dependency를 설치합니다.

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

---

### 2. 전체 실험 실행

전체 실험을 한 번에 실행하려면 다음 명령을 사용합니다.

```bash
bash run_all_experiments.sh
```

이 스크립트는 다음 과정을 자동으로 실행합니다.

```text
1. 4개 PDE dataset 생성
2. 각 PDE에 대해 FNO without coordinate 학습 및 평가
3. 각 PDE에 대해 FNO with coordinate 학습 및 평가
4. 각 PDE에 대해 DeepONet 학습 및 평가
5. 결과를 results/tables/summary.csv에 저장
```

총 실험 조건은 다음과 같습니다.

```text
4 PDE × 3 model settings = 12 runs
```

---

### 3. 개별 실행

#### A. 데이터 생성

```bash
julia --project=. scripts/generate_data.jl configs/periodic_heat.toml
```

#### B. 학습

```bash
# Usage:
# julia --project=. scripts/train_one.jl <config> <model> <use_coord>

julia --project=. scripts/train_one.jl configs/periodic_heat.toml fno false
julia --project=. scripts/train_one.jl configs/periodic_heat.toml fno true
julia --project=. scripts/train_one.jl configs/periodic_heat.toml deeponet false
```

#### C. 평가

```bash
# Usage:
# julia --project=. scripts/evaluate_one.jl <config> <model> <use_coord>

julia --project=. scripts/evaluate_one.jl configs/periodic_heat.toml fno false
julia --project=. scripts/evaluate_one.jl configs/periodic_heat.toml fno true
julia --project=. scripts/evaluate_one.jl configs/periodic_heat.toml deeponet false
```

평가 결과는 다음 파일에 저장됩니다.

```text
results/tables/summary.csv
```

---

## 평가 지표

### Relative L2 Error

기본 예측 성능 지표입니다.

$$\frac{||\hat{u} - u||_2}{||u||_2}$$

값이 낮을수록 예측이 정확합니다.

---

### Initial Sensitivity Error

초기조건이 있는 Heat 계열 문제에서 사용합니다.

서로 다른 초기조건에 대해 정답 해의 차이와 예측 해의 차이가 얼마나 비슷한지 측정합니다.

이 지표는 모델이 초기함수 변화에 따른 출력 변화를 얼마나 잘 보존하는지 확인하기 위한 보조 지표입니다.

---

### Shift Equivariance Error

주기적이고 위치 상대적인 문제에서 사용합니다.

이상적인 translation-relative operator라면 다음 관계가 성립해야 합니다.

```text
G(shift(u)) ≈ shift(G(u))
```

이 지표는 입력을 shift했을 때 모델 출력도 같은 방식으로 shift되는지를 측정합니다.

FNO의 Fourier 기반 구조가 periodic problem에서 shift equivariance를 잘 보존하는지 확인하기 위한 핵심 지표입니다.

---

### Boundary Error

Dirichlet boundary가 있는 문제에서 사용합니다.

예측 해가 경계 조건을 얼마나 잘 만족하는지 측정합니다.

```text
|u_pred(0)| + |u_pred(1)|
```

값이 낮을수록 boundary condition을 더 잘 만족합니다.

---

## Smoke Test

전체 pipeline이 정상적으로 연결되어 있는지 빠르게 확인하려면 smoke test를 실행합니다.

```bash
julia --project=. scripts/smoke_test.jl
```

이 테스트는 작은 dataset과 짧은 epoch 설정으로 다음 과정을 확인합니다.

```text
data generation → training → checkpoint saving → evaluation
```

---

## 기술 스택

- **Language:** Julia
- **Deep Learning:** Lux.jl
- **Neural Operator:** NeuralOperators.jl
- **Automatic Differentiation:** Zygote.jl
- **Optimization:** Optimisers.jl
- **Storage:** JLD2.jl
- **Config:** TOML

---

## 재현성 관련 주의

이 실험은 작은 규모의 1D PDE benchmark입니다.

따라서 결과를 해석할 때 다음 점에 주의해야 합니다.

1. seed 반복 실험을 충분히 수행하지 않았을 수 있습니다.
2. DeepONet은 FNO와 동일한 학습 budget으로 비교되었지만, 충분히 튜닝된 결과는 아닐 수 있습니다.
3. PDE solver와 dataset generation은 실험 목적에 맞춘 단순화된 구현입니다.
4. 결과는 모델의 절대적 우열보다, FNO의 위치 상대성 inductive bias와 coordinate channel의 효과를 관찰하기 위한 것입니다.

---

## AI 사용 고지

이 프로젝트의 구현 코드는 LLM 기반 코딩 도구의 도움을 받아 작성되었습니다.
