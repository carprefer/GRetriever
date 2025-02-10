# G-Retriever report

## 1. Implementation
- evaluate.py, lr.py, gnn.py는 원본 코드를 가져왔다.
- LoRA는 구현하지 않았다.
- 모든 과정에서 모델을 llama-2-7b-chat-hf를 사용하였다.
- 논문에서는 sceneGraphs의 retrieval에서 eCost를 1로 설정하였지만, 원본 코드상에서는 0.5로 설정되어 있었다. 논문을 따라 구현하였다.
- pcst를 돌리고 subGraph를 생성하는 과정에서, 원본 코드는 중복된 node들은 지워주는데, 중복된 edge들은 지워주지 않았다. 의도된 건지는 모르겠으나, 중복된 edge들까지 지워서 구현하였다. 
- 원본 코드에서는 inference-only, prompt-tuning, g-retriever 모두 retrieve를 한 그래프를 사용하는데, 논문에서는 g-retriever에서만 retrieve한 그래프를 사용하므로, 논문을 따라 구현했다.
- 그래프를 textualize할 때, 구분자로 '\n' 대신 '\n\n'을 사용하면 accuracy가 상승한다.

## 2. Evaluation

#### inference-only
```shell
python inference.py --dataset explaGraphs
python inference.py --dataset sceneGraphs
python inference.py --dataset webQsp
```

#### prompt-tuning
```shell
python train.py --dataset explaGraphs --model ptLlm
python train.py --dataset sceneGraphs --model ptLlm
python train.py --dataset webQsp --model ptLlm
```

#### g-retriever
```shell
python train.py --dataset explaGraphs --model graphLlm
python train.py --dataset sceneGraphs --model graphLlm --useGR
python train.py --dataset webQsp --model graphLlm --useGR
```

- 평가 방식을 같게 하기 위하여 evaluate.py는 내려받아 사용하였다.

| Setting | ExplaGraphs | SceneGraphs | WebQsp |
|---------|-------------|-------------|--------|
| Inference-only | 0.5920 | 0.35.72 | 47.02 |
| prompt tuning | 0.5284(0.5126) | 0.48(0.27) | (25)| 
| gRetriever | * | (0.5090) | 61.87 |




## 3. Limitation

- 시간관계상 LoRA는 테스트해보지 못하였다.
- 논문의 결과와 비슷한 양상을 보이지만, 완전히 같지는 않기에 수정이 필요하다.
