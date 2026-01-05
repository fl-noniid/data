# Advanced Adaptive RAG Application

이 프로젝트는 다양한 데이터셋과 RAG 전략을 지원하는 고도화된 Adaptive RAG 애플리케이션입니다.

## 주요 고도화 기능

### 1. 확장된 데이터셋 지원
- **Single-step QA**: SQuAD, Natural Questions (NQ), TriviaQA
- **Multi-step QA**: Musique, HotpotQA, 2WikiMultiHopQA
- **Normal Conversation**: ShareGPT

### 2. 고도화된 검색 및 라우팅
- **Global Corpus**: 데이터셋별 전역 코퍼스를 구축하여 검색 수행
- **Dense Retrieval**: FAISS와 HuggingFace Embeddings를 사용한 밀집 검색
- **Dynamic Routing**: 데이터셋 타입에 따라 최적의 파이프라인 자동 선택
  - `no_retrieval`: LLM 직접 답변 (ShareGPT 등)
  - `single_step`: 1회 검색 후 답변 (SQuAD 등)
  - `multi_step`: IR-CoT 기반 다단계 검색 및 추론 (HotpotQA 등)

### 3. IR-CoT (Interleaving Retrieval and Chain-of-Thought)
Multi-step RAG는 검색과 추론 단계를 교차하여 수행하며, 이전 단계의 추론 결과를 다음 검색의 쿼리로 활용하여 복잡한 질문에 대응합니다.

### 4. 상세 메트릭 측정
각 쿼리별로 다음 지표를 측정하고 출력합니다:
- **End-to-End Latency**: 전체 처리 시간
- **LLM Call Count**: LLM 호출 횟수
- **LLM Latency**: LLM 추론 시간 합계
- **Retrieval Latency**: 검색 시간 합계

## 설치 및 실행

### 의존성 설치
```bash
pip install langchain langchain-community langchain-core openai chromadb datasets sentence-transformers faiss-cpu tabulate tqdm
```

### 실행 예시
```bash
# HotpotQA (Multi-step) 평가
python main.py --dataset hotpotqa --num-samples 10

# SQuAD (Single-step) 평가
python main.py --dataset squad --num-samples 10

# ShareGPT (No-retrieval) 평가
python main.py --dataset sharegpt --num-samples 10
```

## 결과 예시 (출력 화면)
```
+----------+------------+---------------+-------------+---------------+---------------+---------+
| ID       | Strategy   | E2E Latency   | LLM Calls   | LLM Latency   | Ret Latency   | Steps   |
+==========+============+===============+=============+===============+===============+=========+
| 5a8b57f2 | multi_step | 4.231s        | 2           | 3.850s        | 0.381s        | 2       |
+----------+------------+---------------+-------------+---------------+---------------+---------+
```
