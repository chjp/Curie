# Usage instructions
We will run our agent within a docker container.

0. Install docker: https://docs.docker.com/engine/install/ubuntu/

1. Build the container image. Whenever changes have been made: delete the current mounted volume (after backing up necessary data, of course), and rebuild the container image.
```bash
cd Curie/src
conda env create -f environment.yml
sudo docker stop exp-agent-container-instance
sudo docker rm exp-agent-container-instance
sudo docker volume rm exp-agent-container-data
sudo docker build -t exp-agent-image-test -f ExpDockerfile_llm_reasoning_2 ..
```

2. Run and exec into container and begin running experiments. We are using a volume mount to persist the copied agent directory into ``exp-agent-container-data``.
```bash
sudo docker run --cpus=4 --memory=8g --network=host -it --name e2 exp-agent-image-test
conda activate langgraph
source setup/env.sh
python3 main.py configs/base_config.json
# if cloud-infra related questions, use this command INSTEAD, it will essentially add more context to prompt (make sure to populate cloud_helper_related/.aws_creds with the appropriate credentials):
python3 main.py configs/cloud_config.json
python3 main.py configs/llm_reasoning_config.json

# or to run multiple iterations in a loop:
python3 main_loop.py --iterations 1 --pipeline curie --timeout 600 --category vdb
python3 main_loop.py --iterations 2 --pipeline curie --timeout 600 --category vdb --questions_to_run q14
python3 main_loop.py --iterations 5 --pipeline openhands --timeout 600 --category reasoning2 --questions_to_run q5 q8 q10
python3 main_loop.py --iterations 5 --pipeline magentic --timeout 600 --category mltraining --questions_to_run q1

# we can even call the parallel runner which will start a subprocess for each individual question to test:
python3 parallel_runner_main_loop.py --config configs/parallel_run_config.json

# in another terminal, you may consider viewing redirected stdout: 
sudo tail -f /var/lib/docker/volumes/exp-agent-container-data/_data/misc/log-temp.log
```

3. (optional) You can also exec into the container using this command.
```bash
sudo docker exec -it exp-agent-container-instance bash
```

4. (optional) Occasionally, you may want to view or delete the contents of the mounted volume: 
```bash
sudo ls /var/lib/docker/volumes/exp-agent-container-data/_data/

sudo rm -rf /var/lib/docker/volumes/exp-agent-container-data/_data/

# Actually this command is sufficient: skip the above 
sudo docker volume rm exp-agent-container-data
```

5. (optional) You may want to open or copy a log file without sudo:
```bash
# Copies all files with the pattern below, into a new folder within our host:
sudo find /var/lib/docker/volumes/exp-agent-container-data/_data/logs/ -type f -name "log-temp[0-9]*.log" -exec cp {} logs/ \;
# Update: for some reason it is better to use docker cp directly, sometimes volume does not reflect latest change.... (only noticed this for mass copies):
sudo docker exec exp-agent-container-20250122014939-iter1 sh -c 'find /temp/logs -maxdepth 1 -type f -name "*.log" | tar -cf - -T -' | tar -xf - -C misc/logs
```

6. (optional) If you run out of disk space:
```bash
sudo docker system df
# Do the following as needed:
sudo docker container prune
sudo docker image prune
sudo docker volume prune
# This usually helps the most:
sudo docker builder prune
```