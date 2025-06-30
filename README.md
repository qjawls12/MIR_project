# MIR_project
# 사용자 친화적 보컬 트레이닝 시스템

실시간 피드백 및 사후 분석 기능을 갖춘 웹 기반 보컬 트레이닝 시스템입니다. 이 프로젝트는 보컬 연습의 효율성을 높이기 위해 사용자 피치 정보를 실시간으로 분석하고, 정답 멜로디와의 차이를 직관적으로 시각화하는 기능을 제공합니다.

## 📌 프로젝트 개요

- **목표**: 사용자의 노래 실력을 향상시키기 위한 실시간 피드백 및 사후 분석 기반의 보컬 트레이닝 시스템 개발
- **주요 기능**:
  - 실시간 피치 추정 (CREPE 기반)
  - 정답 멜로디와의 비교 시각화
  - GUI 기반 직관적인 사용 경험 제공
  - 가사 동기화 및 피드백 표시
  - 추후 사후 분석 기능 확장 (예정)

## 실시간 피치 추정 및 시각화 시스템 구조

```
[마이크 입력] → [CREPE 피치 추정] → [정답 멜로디 비교] → [실시간 시각화 & 피드백 표시]
                                 ↘ [가사 동기화] ↘
                                  [PyQt5 기반 GUI]
```

## Step 1. Source Seperation
음원 분리 모델 Demucs를 활용하여, 음원에서 보컬과 배경음을 분리합니다.
재생바를 통해 분리가 진행되는 동안 음원을 들어볼 수 있습니다.


[![Demo Video](https://img.youtube.com/vi/zmPvw4T5XjY/0.jpg)](https://www.youtube.com/watch?v=zmPvw4T5XjY)


## Step 2. Lyric Extraction
음성 인식 모델 Whisper을 사용해, 추출한 보컬 음원에서 가사를 추출합니다.
실시간으로 추출되는 가사를 확인할 수 있습니다.


[![Demo Video](https://img.youtube.com/vi/umKj_Yq5YTg/0.jpg)](https://www.youtube.com/watch?v=umKj_Yq5YTg)

## Step 3. (Optional) Key Transposition

음성 처리 모듈 rubberband를 통해, 원하는만큼 키를 높이고 낮출 수 있습니다.
실시간으로 피치가 바뀌는 음성을 확인할 수 있습니다.

[![Demo Video](https://img.youtube.com/vi/gGakj3l56_o/0.jpg)](https://www.youtube.com/watch?v=gGakj3l56_o)

## Step 4. Pitch Extraction

Pitch Estimation 모듈 torchcrepe를 통해, 가이드 보컬 음원에서 정답 피치값을 추출합니다.
추출된 정보는 npy 형태로 저장되어, 다음 단계에서 활용됩니다.

[![Demo Video](https://img.youtube.com/vi/8QpaUfXije0/0.jpg)](https://www.youtube.com/watch?v=8QpaUfXije0)

## Step 5. Real-time Vocal Training

사용자의 음원을 실시간으로 입력받고, 피치를 추출한 뒤, 정답 피치와 실시간으로 비교하여 시각화합니다.
일치 여부를 색깔로 표현하고, 점수화하여 표현합니다. 화면 하단에서 가사를 확인할 수 있습니다.

[![Demo Video](https://img.youtube.com/vi/ffU5fSqf4jw/0.jpg)](https://www.youtube.com/watch?v=ffU5fSqf4jw)
