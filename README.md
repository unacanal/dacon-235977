# **AI 양재 허브 인공지능 오픈소스 경진대회**

## Descriptions
### [주제]
이미지 초해상화(Image Super-Resolution)를 위한 AI 알고리즘 개발
### [설명]
품질이 저하된 저해상도 촬영 이미지(512X512)를 고품질의 고해상도 촬영 이미지(2048X2048)로 생성
### [홈페이지]
[Link](https://www.dacon.io/competitions/official/235977/overview/description)


## About our team
> Team        : Super <p>
Member        : Super, 우주대마왕, 이여름, MingiHong, 쾌남 <p>
Public Score  : 25.35969 (5th) <p>
Private Score : 24.23838 (8th)


## Installation & Setting
This implementation is based on [HAT](https://github.com/XPixelGroup/HAT) and [BasicSR](https://github.com/XPixelGroup/BasicSR). 

```python
python 3.9.13
pytorch 1.12.1
cuda 11.3
```

```
pip install -r requirements.txt
python setup.py develop
```

## Data preparation
[data/README.md](https://github.com/unacanal/dacon-235977/tree/master/data) 참고

## Pretrained models
사전 학습된 모델을 ```experiments/pretrained_models``` 디렉토리에 위치

|         Pretrained Model         |                           제공                             |
|:--------------------------------:|------------------------------------------------------------|
| HAT-L_SRx4_ImageNet-pretrain.pth | [HAT GitHub](https://github.com/XPixelGroup/HAT)                                         |
| HAT-L_Seoul_115K.pth             | 서울 데이터셋으로 학습 또는 [다운로드](https://drive.google.com/drive/folders/1nu9UvbKnNeaa6dHRotW5CNBk8d-XXDtC?usp=sharing)           |
| HAT-L_DACON_175K.pth             | 데이콘 챌린지 제공 데이터셋으로 학습 또는 [다운로드](https://drive.google.com/drive/folders/1nu9UvbKnNeaa6dHRotW5CNBk8d-XXDtC?usp=sharing) |ㄴ

## Train/Test
*학습 결과는 ```experiments/```, 테스트 결과는 ```results/``` 폴더 아래에 저장됨*
- Train with Seoul dataset
    ```
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train_HAT-L_Blind_Seoul_from_ImageNet_pretrain.yml --launcher pytorch
    ```
- Train with DACON dataset (After training with Seoul dataset)
    ```
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train_HAT-L_DACON_from_Seoul_pretrain.yml --launcher pytorch
    ```

- Test (**Private score 재현**)
    ```
    python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/test.py -opt options/test_HAT-L_DACON_from_Seoul_pretrain.yml --launcher pytorch
    ```
