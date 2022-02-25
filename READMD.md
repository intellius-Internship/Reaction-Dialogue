# **Reaction-Dialogue**


### **파일 구조**

```bash
.
├── data    
│   ├── kw_based/               키워드 기반 labeled dialogue dataset
│   ├── regexp_based/           정규식 매칭 기반 labeled dialogue dataset
│   ├── textdist_based/         텍스트유사도 기반 labeled dialogue dataset
│   │   
│   ├── DATA.md                
│   ├── data.csv                Raw Dialogue Dataset (약 300만건의 싱글턴 대화)
│   ├── pingpong_reaction.csv   핑퐁빌더에서 공유한 리액션 분류
│   └── reaction.csv            Reaction-RegExp Dataset, 임의로 선정한 115개 리액션 클래스의 키워드/정규식
│
├── preprocessing               데이터 라벨링 및 전처리 
│   ├── ...
│   ├── build_dataset.py        데이터셋 구축을 위한 실행 코드
│   └── ...                 
│
├── result/                     모델 테스트 결과 저장 경로
├── utils/
├── ...
├── main.py                     모델 학습 및 테스트를 위한 실행 코드
├── READMD.md
└── ...
```

<br>


## **Building Reaction Dataset** 


```bash
cd preprocessing/
```

### 1. Reaction Labeling

- `labeling`: 라벨링 메소드 (default=None)   
    - `keyword` : 키워드 기반 라벨링
    - `textdist` : 텍스트 유사도 기반 라벨링
    - `regexp` : 정규식 매칭 기반 라벨링

```bash
python build_dataset.py --labeling regexp --data_dir ../data --result_dir ../result
```

### 2. Build Training, Validation, Test dataset
```bash
python build_dataset.py --preprocessing --split --data_dir ../data --result_dir ../result
```

<br>

---

## **Training/Testing Reaction Dialogue Model** 

<br>

- `model_type`: 모델 유형      
    - `gpt2` : Pretrained KoGPT2 (`skt/kogpt2-base-v2`)
    - `bart` : Pretrained KoBART (`gogamza/kobart-base-v2`)

### 1. Training

```bash
python main.py --train --max_epochs 10 --data_dir data/regexp_based --model_type gpt2 --model_name gpt2_chat --max_len 64 --gpuid 0
```

<br>

### 2. Testing

*하나의 GPU만 사용*  

#### (1) `<data_dir>`/test.csv에 대한 성능 테스트

```bash
python main.py --data_dir data/regexp_based --model_type gpt2 --model_name gpt2_chat --save_dir result --max_len 64 --gpuid 0 --model_pt <model checkpoint path>
```

#### (2) 사용자 입력에 대한 성능 테스트

```bash
python main.py --chat --data_dir data/regexp_based --model_type gpt2 --max_len 64 --gpuid 0 --model_pt <model checkpoint path>
```

<br>


