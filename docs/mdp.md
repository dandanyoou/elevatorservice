# MDP 정의: Multi-Elevator Scheduling

이 문서는 `envs/elevator_env.py`가 구현하는 MDP의 정식 정의이다. RL 학습 시
state / action / reward / episode가 어떻게 잡혀 있는지를 한 곳에서 명세한다.

## 1. 시스템 파라미터

`envs/config.py`의 `BuildingConfig`로 노출:

| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `num_floors` (F) | 10 | 층수 (0층 = 로비) |
| `num_elevators` (N) | 3 | 엘리베이터 대수 |
| `capacity` (C) | 12 | 1대당 정원 |
| `floor_height_m` | 3.5 | 층고 |
| `speed_mps` | 2.0 | 정속 주행 속도 |
| `door_open_seconds` | 2.0 | 문 열림 시간 |
| `door_close_seconds` | 2.0 | 문 닫힘 시간 |
| `boarding_seconds_per_passenger` | 1.0 | 1인 승하차 시간 |
| `dt_seconds` | 1.0 | 시뮬 1 step 길이 |
| `horizon_steps` | 3600 | 1 에피소드 = 1시간 |

## 2. 상태 (State)

관측 벡터는 길이 `obs_dim = N·(F+3) + 2F + 1` 의 `Box(-1, 1)`. 의미적으로는
다음을 평탄화한 것:

### 엘리베이터 i ∈ {1..N} 별
1. 현재 층 (one-hot, 길이 F): `floor_i`
2. 방향 (스칼라, ∈ {-1, 0, +1}): `dir_i`
3. 부하율 (스칼라, ∈ [0, 1]): `passengers_i / C`
4. 서비스 중 여부 (스칼라, ∈ {0, 1}): `servicing_i`

### 외부 호출 (Hall calls)
5. 상승 호출 (binary, 길이 F): `hall_up`
6. 하강 호출 (binary, 길이 F): `hall_down`

### 시간 컨텍스트
7. 시간 진행률 (스칼라, ∈ [0, 1]): `step_count / horizon`

내부 호출 (탑승객 목적지)은 **부하율 + 서비스 플래그**로 압축되어 있다. 학습 시
필요하면 internal_call 마스크를 별도 채널로 추가하는 것이 자연스러운 확장점이다
(현재는 학습 신호가 충분한지부터 확인하는 단계).

## 3. 행동 (Action)

`MultiDiscrete([3] * N)` — 엘리베이터 i 마다:

| 값 | 의미 |
|---|---|
| 0 | DOWN (한 층 아래로 이동 시도) |
| 1 | HOLD (현재 위치 유지 / 정지) |
| 2 | UP (한 층 위로 이동 시도) |

### 환경의 자동 처리
다음 케이스는 행동을 무시하거나 강제 정지로 해석한다:
- 서비스 중(`is_servicing == True`): 행동 무시
- 운행 중인 방향과 반대 명령: 다음 층 도착 시 정지
- 도착한 층에 같은 방향 hall call 또는 내부 목적지가 있음: 강제 정지·서비스
- 종착층 도달: 강제 정지

이 자동 처리 덕분에 ``HOLD``를 의미 있는 결정 신호로 쓸 수 있다 (학습 정책 입장:
"이 층에서 잠깐 쉬어"). 또한 baseline (SCAN, Nearest-Car)이 깔끔하게 표현된다.

## 4. 보상 (Reward)

매 step `t`에서:

```
r_t = - α · Σ(이번 step 동안 대기 중이던 인원 수) · dt
      - β · Σ(이번 step 동안 차내 인원 수) · dt
      - γ · (이번 step 총 운행 층-equivalent)
      - δ · (이번 step 발생한 방향 전환 횟수)
      + ε · (이번 step 도착 완료 인원 수)
```

기본 가중치 (`RewardWeights` dataclass):

| 기호 | 키 | 기본값 |
|---|---|---|
| α | `waiting` | 0.01 |
| β | `in_car` | 0.005 |
| γ | `energy` | 0.001 |
| δ | `direction_change` | 0.1 |
| ε | `completion_bonus` | 1.0 |

설계 의도:
- α/β: AWT, AJT (Average Wait/Journey Time) 와 직접 연결되는 누적 부담
- γ: 에너지 페널티 — 무의미한 왕복을 억제
- δ: 부드러운 운행 (잦은 방향 전환은 승차감·기계 부하 모두 나쁨)
- ε: 도착 시 양의 신호. α/β의 음수와 균형을 잡아 sparse-reward 회피

가중치는 step-level cost에 맞춰 작게 잡혀 있다. 1시간(3600 step) 에피소드 기준
순수 cost는 보통 -O(50~200), completion은 도착 인원만큼 +. PPO에서 reward
clipping/normalization이 필요하면 학습 단계에서 추가한다.

## 5. 트래픽 모델

`TrafficConfig.pattern` ∈ {`uniform`, `morning`, `lunch`, `evening`}:
- `uniform`: 모든 (origin, dest) 페어 동일 가중치
- `morning`: 로비(0층)→상위층 6배, 그 외 0.3배
- `lunch`: 로비/중간층 source/sink 4배
- `evening`: 상위층→로비 6배

도착은 step별 Poisson(λ), 1분당 평균 `base_rate_per_minute`. 한 시간 기준 8 ppm =
약 480명이 디폴트 부하.

## 6. 에피소드

- 길이: `horizon_steps` (기본 3600 step = 1시간)
- 종료: truncation only (terminated 항상 False — 정상 운영 시 자연 종료가 없는
  연속 시스템)
- `metrics()` 호출 시 도착 완료자 기준 AWT, AJT, throughput 반환

## 7. 결정론성 / 재현성

`reset(seed=k)` 시 traffic generator를 시드 k로 초기화. 같은 seed + 같은
action 시퀀스 → 동일한 obs/reward 보장. `tests/test_env.py::test_determinism_under_fixed_seed`
가 이 속성을 검증한다.

## 8. 향후 확장 포인트

- **action space 확장**: target floor (Discrete F) + idle 토큰의 semi-MDP
  버전. 현재 v1은 step-단위 단순 명령으로 시작했다.
- **action mask**: `is_servicing` 동안의 step에서 효과 없는 행동을 mask 형태로
  policy에 노출 (`gymnasium.spaces.Dict`로 obs 확장).
- **multi-agent**: 엘리베이터별 별도 agent (Ray RLlib `MultiAgentEnv`).
- **partial observability**: hall_up/down 직접 노출 대신 “목격된 호출"만 노출.
- **에너지 모델 정교화**: 가속/감속에 따라 비대칭 비용. 현재는 운행 거리에
  비례.
