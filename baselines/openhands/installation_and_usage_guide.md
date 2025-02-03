# OpenHands installation and usage guide

## Installation steps:
1. git clone openhands repository
2. Just follow the steps here: https://docs.all-hands.dev/modules/usage/how-to/headless-mode

3. Fill in the details in ```config.toml``` and copy the file to the openhands repo base directory: 
```bash
cp langgraph-exp-agent/baselines/openhands/config.toml /home/patkon/OpenHands/openhands
```

4. To solve the following error (may encounter when running openhands for the first time):
```bash
ERROR:root:<class 'ImportError'>: /home/patkon/miniconda3/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.32' not found (required by /home/patkon/.cache/pypoetry/virtualenvs/openhands-ai-mCt_2w7a-py3.12/lib/python3.12/site-packages/_pylcs.cpython-312-x86_64-linux-gnu.so)
```
- Use this solution:
```bash
ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ${CONDA_PREFIX}/lib/libstdc++.so.6
```

## Usage steps for main_loop.py:

### Clone repos into workspace/
```bash
cd baselines/openhands
mkdir -p workspace/llm_reasoning_related/starter_file
cd workspace/llm_reasoning_related/starter_file
git clone https://github.com/AmberLJC/large_language_monkeys
# follow instructions in the README.md under benchmark/llm_reasoning which will include details on env.sh for instance

# And also for vector db
mkdir -p workspace/vector_index_related/starter_file
cd workspace/vector_index_related/starter_file
git clone https://github.com/patrickkon/faiss.git
```

### Run main_loop.py
Follow instructions in proto1/README.md

## Usage steps:

### Set env variables:
1. ```export LOG_ALL_EVENTS=true```
- https://github.com/All-Hands-AI/OpenHands/issues/4819
2. ```export SANDBOX_TIMEOUT=600```
- https://github.com/All-Hands-AI/OpenHands/issues/4877 set this to whatever you need. 

### CD into openhands dir:
```cd OpenHands```

### Provide a file containing the task (default for us):
1. ```poetry run python -m openhands.core.main -f /home/patkon/langgraph-exp-agent/baselines/openhands/prompt_cloud_q2b.txt 2>&1 | tee -a /home/patkon/langgraph-exp-agent/baselines/openhands/logs/log-1-q2b.txt```
2. Alternative command ```poetry run python -m openhands.core.main -f /home/patkon/langgraph-exp-agent/benchmark/cloud_infra/t3_c5_cpu_workload_ques/baseline_question_format/question-level-2.txt 2>&1 | tee -a /home/patkon/langgraph-exp-agent/baselines/openhands/logs/t3_c5_cpu_workload_ques/log-level-2-1.txt```
- we will redirect stdout and stderr to a log file

### Provide a sentence containing the task:
1. ```poetry run python -m openhands.core.main -t "write a bash script that prints hi"```

### Reminders:
- Note: don't use the original command poetry run ```python -m openhands.core.main -t "write a bash script that prints hi" --no-auto-continue```, as that will wait for user input. 