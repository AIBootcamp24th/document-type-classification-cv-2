[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/6L7Rprz9)

# 문서 이미지 분류 대회 (Document Type Classification)

## 팀원
- 송병찬
- 유준우
- 이수지
- 이재석(팀장)
- 최유정

## 대회 개요
이 대회는 다양한 문서 이미지 데이터를 바탕으로 문서 타입을 분류하는 컴퓨터 비전 경진대회입니다. 참가자는 모델 학습부터 평가에 이르는 전체 프로세스를 경험하며 모델 개선 과정을 학습하는 것을 목표로 합니다.

## 데이터 구성
### 학습 데이터
- 총 17개의 클래스
- 총 1570장의 이미지 (클래스당 46~100장)

### 테스트 데이터
- 총 3140장의 이미지
- 구겨짐, 빛번짐 등 다양한 노이즈 포함
- 난이도 조절을 위한 여러 augmentation 적용

## 평가 방법
- **성능 평가 지표**: Macro F1 score
- **분석 도구**: Confusion Matrix, Precision(정밀도), Recall(재현율) 활용
