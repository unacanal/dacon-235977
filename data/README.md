## Download Dataset
1. DACON challenge dataset [Link](https://drive.google.com/file/d/17Ui8Pc6NiPTd6dBsZJr8XS4l9S9F-wyL/view)
2. 서울 데이터셋 [Link](https://data.seoul.go.kr/etc/aiEduData.do)

### 디렉토리 구조

    ```
    data
    ├── dacon
    │   ├── train
    │   │   ├── lr
    │   │   └── hr
    │   └── test
    │       └── lr
    |
    ├── seoul
    |    └── hr
    |
    └── create_lmdb.py
    ```

모든 학습 데이터는 빠른 입출력을 위해 lmdb 포맷으로 변환

```
python create_lmdb.py --dataset dacon
python create_lmdb.py --dataset seoul
```


