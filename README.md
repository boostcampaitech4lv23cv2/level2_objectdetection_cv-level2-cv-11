# level2_objectdetection_cv-level2-cv-11
![image](https://user-images.githubusercontent.com/30896956/205215342-d1431a61-4228-492c-a9d8-0196bdf76556.png)


## 프로젝트 개요

 사진에서 쓰레기를 Detection 하는 모델을 만들어 잘못 분리배출되는 문제점을 해결해보고자 합니다. 문제 해결을 위한 데이터셋으로는 일반 쓰레기, 플라스틱, 종이, 유리 등 10 종류의 쓰레기가 찍힌 사진 데이터셋이 제공됩니다.

 - Input : 쓰레기 객체가 담긴 이미지와 COCO format의 bbox annotation
 - 전체 데이터셋 중 약 50%(4883장)는 학습 데이터셋으로 활용됩니다.
 - Output : 모델은 bbox 좌표, 카테고리, score 값을 리턴합니다. 이를 submission 양식에 맞게 csv 파일을 만들어 제출합니다. (submission format에 대한 설명은 평가방법을 참고해주세요.)

Evaluation

- mAP50(Mean Average Precision)로 평가됩니다.
![image](https://user-images.githubusercontent.com/62556539/205196213-21d58e35-e92f-463a-9d32-41954ad0dbef.png)

## 프로젝트 팀 구성

| [류건](https://github.com/jerry-ryu) | [심건희](https://github.com/jane79) | [윤태준](https://github.com/ta1231) | [이강희](https://github.com/ganghe74) | [이예라](https://github.com/Yera10) |
| :-: | :-: | :-: | :-: | :-: | 
| <img src="https://avatars.githubusercontent.com/u/62556539?v=4" width="200"> | <img src="https://avatars.githubusercontent.com/u/48004826?v=4" width="200"> | <img src="https://avatars.githubusercontent.com/u/54363784?v=4"  width="200"> | <img src="https://avatars.githubusercontent.com/u/30896956?v=4" width="200"> | <img src="https://avatars.githubusercontent.com/u/57178359?v=4" width="200"> |  
|[Blog](https://kkwong-guin.tistory.com/)  |[Blog](https://velog.io/@goodheart50)|[Blog](https://velog.io/@ta1231)| [Blog](https://six-six.notion.site/six-six/3435d13b28c84193aeacb8d50dcdd239?v=b48a26ab21a745819220b7618089fc93) | [Blog](https://yedoong.tistory.com/) |

<div align="center">

![python](http://img.shields.io/badge/Python-000000?style=flat-square&logo=Python)
![pytorch](http://img.shields.io/badge/PyTorch-000000?style=flat-square&logo=PyTorch)
![ubuntu](http://img.shields.io/badge/Ubuntu-000000?style=flat-square&logo=Ubuntu)
![git](http://img.shields.io/badge/Git-000000?style=flat-square&logo=Git)
![github](http://img.shields.io/badge/Github-000000?style=flat-square&logo=Github)

</div align="center">

## 디렉토리 구조

```CMD
level2_objectdetection_cv-level2-cv-11
└── baseline
    └── UniverseNet   # MMDetection 기반 패키지
    └── detectron2
    └── detrex        # Detectron2 기반 패키지
    └── faster_rcnn   
    └── mmdetection
    └── yolov7        # 자체 라이브러리
└── notebooks         # EDA, ensemble, pseudo labeling 등


```


## 프로젝트 수행 환경
모든 실험은 아래의 환경에서 진행되었다.
- Ubuntu 18.04.5 LTS
- Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz
- NVIDIA Tesla V100-PCIE-32GB


## 프로젝트 수행 절차 및 방법

[![image](https://user-images.githubusercontent.com/62556539/200262300-3765b3e4-0050-4760-b008-f218d079a770.png)](https://excessive-help-ce8.notion.site/b41d8c9e26b64e8f9e18b529ebc1f287)



## 프로젝트 수행 결과

![image](https://user-images.githubusercontent.com/62556539/205196491-ecdcabd1-e4eb-4ff8-8607-61205d9b4ac6.png)

![image (6)](https://user-images.githubusercontent.com/48004826/205533073-b9e1c4ec-2fb0-4653-8097-f8a923e53719.png)


최종 모델: 1stage, 2 stage 각각 앙상블 후 다시 앙상블


## 자체 평가 의견

**잘한 점**
- Github Convention을 정하여 협업 결과 공유 및 정리에 큰 도움이 되었다.  
- Github 이슈, PR 등의 기능을 적극적으로 사용해보았다.  
- 힘든 일정에도 서로를 격려하고 팀 분위기를 긍정적으로 유지하였다.  
- 다양한 SOTA object detector 에 대해 공부하고 실험했다.  
- 함께 디버깅하여 빠른 문제 대응을 할 수 있었다.  
- PR-Curve 등 Wandb 기능을 커스텀하여 다양한 메트릭을 볼 수 있었다.  
- Wandb를 사용하여 실시간 모니터링 및 팀원들과의 결과 공유가 용이했다.  

**아쉬운 점:**
- 추가한 코드에 대해 코드 내에서 설명이 부족했다.
- 모델 및 기법들에 대해 이론적인 공부와 결과분석이 부족했다.
- 대회 종료까지 Bounding Box의 크기, 비율을 고려하지 않은 데이터셋 분할을 사용해서 모델 학습과 검증이 잘 안되었다.
- augmentation 에 대한 실험이 부족했다.
- Kaggle competition 의 토론글들을 모델 외의 method 선정에 활용하지 못했다.
- 팀원간의 방향성 공유가 부족했다.
- data 근거에 기반해서 실험계획이 수립되지 않았다.

**개선할 점:**
- 주어진 시간 내에 다양한 변수를 모두 고려하여 균형 잡힌 실험 계획을 할 것이다.
- Kaggle competition의 인기 토론글들을 좀 더 적극 활용 해볼 것이다.
- 팀 적 활동을 더 많이 하도록 노력해야겠다.
- 팀 내의 문제점과 목표를 서로 공유하고 다같이 다듬어 나갈 수 있는 방안을 강구해야겠다
- 쉬운 것부터 단계적으로 실험해나갔으면 좋겠다.
---
