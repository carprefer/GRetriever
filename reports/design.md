# G-Retriever Design

[Language] Python 3.9

[Os] Linux server(dslab)

[GPU] Tesla P40

[CUDA SDK] 12.4

[PyTorch] 2.5.1

[LLM] LLaMA 2-7b-hf

## Milestone
1. 환경 설정
2. 데이터 전처리
3. indexing module 구현 및 테스트(embedding 생성)
4. Inference-only model 구현 및 테스트
5. retrieval module 구현 및 테스트
6. subgraph construction module 구현 및 테스트
7. answer generation module 구현 및 테스트
8. 각 데이터셋에 대하여 training 및 평가 진행

## Environment Setup

#### miniconda 가상환경 설정
```shell
conda create --name gRetriever python=3.9 -y
conda activate gRetriever

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

pip install transformer
pip install pandas
pip install accelerate
pip install datasets
pip install torch_geometric
```

## LLM 설정
```python
from transformers import pipeline

pipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-hf")
```

## Dataset 다운로드

#### ExplaGraphs
- explaGraphs에 대해서는 따로 retrieve 과정이 필요없다.

#### SceneGraphs
- train_sceneGraphs.json과 questions.csv만 사용한다.

#### WebQSP
```python
from datasets import load_dataset

ds = load_dataset("rmanluo/RoG-webqsp")
```

