## Setup for docker container running:
We want our magentic to be able to access root
```
cd langgraph-exp-agent/proto1
sudo docker build --build-arg PROMPT_FILE=/benchmark/ml_training/q1_housing_price.txt -t magentic-agent-image -f MagenticDockerfile ..

docker build --cache-from magentic-agent-image:latest -t magentic-agent-image -f MagenticDockerfile .. 

docker run -v /var/run/docker.sock:/var/run/docker.sock -v /usr/bin/docker:/usr/bin/docker  --cpus=4 --memory=8g --network=host -it --name test1 magentic-agent-image

```
cd ../../
cd langgraph-exp-agent/starter_file
git clone https://github.com/AmberLJC/autogen.git

## Setup
```
cd $HOME
git clone https://github.com/AmberLJC/autogen.git
pip install -U "autogen-agentchat" "autogen-ext[openai]"

cd autogen/python
pip install uv
uv sync  --all-extras
source .venv/bin/activate

cd packages/autogen-magentic-one
pip install -e .
```
## Setup the API key
Create a new file in $HOME/langgraph-exp-agent/proto1/setup/env.sh:
```
export MODEL="gpt-4o"
export API_VERSION="2024-06-01"
export OPENAI_API_KEY='?'
export OPENAI_ORGANIZATION='?'
export OPENAI_API_BASE='https://api.umgpt.umich.edu/azure-openai-api'
```

source $HOME/langgraph-exp-agent/proto1/setup/env.sh

## Run benchmark questions
```
cd $HOME
export PROMPT_FILE='langgraph-exp-agent/benchmark/ml_training/q1_housing_price.txt'
export STARTER_FILES='MLAgentBench/MLAgentBench/benchmarks/house-price/'

playwright install
playwright install-deps
python /starter_file/autogen/python/packages/autogen-magentic-one/examples/example.py  --logs_dir $STARTER_FILES

```
## Notes:
- Ensure that docker can run without sudo

You can change the resource budget by configuring `LedgerOrchestrator` in `example.py`.

