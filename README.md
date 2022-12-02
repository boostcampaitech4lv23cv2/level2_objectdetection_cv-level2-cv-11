# level2_objectdetection_cv-level2-cv-11
![image](https://user-images.githubusercontent.com/30896956/205215342-d1431a61-4228-492c-a9d8-0196bdf76556.png)


## 프로젝트 개요

바야흐로 대량 생산, 대량 소비의 시대. 우리는 많은 물건이 대량으로 생산되고, 소비되는 시대를 살고 있습니다. 하지만 이러한 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있습니다.

분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문입니다.

따라서 우리는 사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 문제 해결을 위한 데이터셋으로는 일반 쓰레기, 플라스틱, 종이, 유리 등 10 종류의 쓰레기가 찍힌 사진 데이터셋이 제공됩니다.

여러분에 의해 만들어진 우수한 성능의 모델은 쓰레기장에 설치되어 정확한 분리수거를 돕거나, 어린아이들의 분리수거 교육 등에 사용될 수 있을 것입니다. 부디 지구를 위기로부터 구해주세요! 🌎

 - Input : 쓰레기 객체가 담긴 이미지가 모델의 인풋으로 사용됩니다. 또한 bbox 정보(좌표, 카테고리)는 model 학습 시 사용이 됩니다. bbox annotation은 COCO format으로 제공됩니다. (COCO format에 대한 설명은 학습 데이터 개요를 참고해주세요.)

 - Output : 모델은 bbox 좌표, 카테고리, score 값을 리턴합니다. 이를 submission 양식에 맞게 csv 파일을 만들어 제출합니다. (submission format에 대한 설명은 평가방법을 참고해주세요.)

Evaluation

Test set의 mAP50(Mean Average Precision)로 평가

Object Detection에서 사용하는 대표적인 성능 측정 방법
Ground Truth 박스와 Prediction 박스간 IoU(Intersection Over Union, Detector의 정확도를 평가하는 지표)가 50이 넘는 예측에 대해 True라고 판단합니다.

![image](https://user-images.githubusercontent.com/62556539/205196213-21d58e35-e92f-463a-9d32-41954ad0dbef.png)

## 프로젝트 팀 구성

| [류건](https://github.com/jerry-ryu) | [심건희](https://github.com/jane79) | [윤태준](https://github.com/ta1231) | [이강희](https://github.com/ganghe74) | [이예라](https://github.com/Yera10) |
| :-: | :-: | :-: | :-: | :-: | 
|![image](https://avatars.githubusercontent.com/u/62556539?v=4)|![image](https://avatars.githubusercontent.com/u/48004826?v=4)|![image](https://avatars.githubusercontent.com/u/54363784?v=4) |![image](https://avatars.githubusercontent.com/u/30896956?v=4)  |![image](https://avatars.githubusercontent.com/u/57178359?v=4)|  
|[Blog](https://kkwong-guin.tistory.com/)  |[Blog](https://velog.io/@goodheart50)|[Blog](https://velog.io/@ta1231)| Blog | [Blog](https://yedoong.tistory.com/) |

<div align="center">

![python](http://img.shields.io/badge/Python-000000?style=flat-square&logo=Python)
![pytorch](http://img.shields.io/badge/PyTorch-000000?style=flat-square&logo=PyTorch)
![ubuntu](http://img.shields.io/badge/Ubuntu-000000?style=flat-square&logo=Ubuntu)
![git](http://img.shields.io/badge/Git-000000?style=flat-square&logo=Git)
![github](http://img.shields.io/badge/Github-000000?style=flat-square&logo=Github)

</div align="center">

## 디렉토리 구조

```CMD

```


## 프로젝트 수행 환경
모든 실험은 아래의 환경에서 진행되었다.
- Ubuntu 18.04.5 LTS
- Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz
- NVIDIA Tesla V100-PCIE-32GB


## 프로젝트 수행 절차 및 방법

[![image](https://user-images.githubusercontent.com/62556539/200262300-3765b3e4-0050-4760-b008-f218d079a770.png)](https://www.notion.so/3f3a9a603c5348939bd3c16996aa2c22)



## 프로젝트 수행 결과

![image](https://user-images.githubusercontent.com/62556539/205196491-ecdcabd1-e4eb-4ff8-8607-61205d9b4ac6.png)


최종 모델:  앙상블


## 자체 평가 의견

**잘한 점**

**아쉬운 점:**

**개선할 점:**

---
