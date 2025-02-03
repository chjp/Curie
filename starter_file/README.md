## Initialize required repos for evaluation

### Cloud infra:
Create a .aws_creds file for both best_instance/ and cpu_workload/sysbench_workload, with content like this:
```
[default]
aws_access_key_id=???
aws_secret_access_key=???
region=us-east-1
```

### Faiss for VDB tasks:
Follow instructions in langgraph-exp-agent/benchmark/vector_index/README.md

### Large language monkeys for LLM reasoning tasks:
Follow instructions in langgraph-exp-agent/benchmark/llm_reasoning/README.md

### Scaling for LLM reasoning 2 tasks:
Follow instructions in langgraph-exp-agent/benchmark/llm_reasoning_2/README.md

### MLAgentBench for ML training tasks:
Follow instructions in langgraph-exp-agent/benchmark/ml_training/README.md

