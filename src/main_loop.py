import subprocess
import time
import os
import shutil  # Import for deleting directories
import psutil
import argparse
from enum import Enum
import json
import re
from datetime import datetime 
import sys
import uuid

# Define an Enum for the baseline values
class Pipeline(Enum):
    OPENHANDS = "openhands"
    CURIE = "curie"
    MAGENTIC = "magentic"

    def __str__(self):
        return self.value  # For proper display in argparse help text

# Define an Enum for the question category values
class QCategory(Enum):
    REASONING = "reasoning"
    VDB = "vdb"
    CLOUD = "cloud"
    MLTRAINING = "mltraining"
    REASONING2 = "reasoning2"

    def __str__(self):
        return self.value  # For proper display in argparse help text

# Create a function to parse input arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Process input arguments for the script.")
    
    # Add argument for iterations
    parser.add_argument(
        "--iterations",
        type=int,
        required=True,
        help="Number of iterations (must be an integer)."
    )
    
    # Add argument for baseline
    parser.add_argument(
        "--pipeline",
        type=Pipeline,
        choices=list(Pipeline),
        required=True,
        help="Pipeline value (must be one of: openhands, curie)."
    )

    # Add argument for question category
    parser.add_argument(
        "--category",
        type=QCategory,
        choices=list(QCategory),
        required=True,
        help="Question category (must be one of: reasoning, vdb, cloud, mltraining, reasoning2)."
    )
    
    # Add argument for timeout
    parser.add_argument(
        "--timeout",
        type=int,
        required=True,
        help="Timeout in seconds (must be an integer)."
    )

    # Add argument for questions_to_run, default to None, otherwise if provided, it should be of type list:
    parser.add_argument(
        "--questions_to_run",
        nargs="*", # All command-line arguments present are gathered into a list.
        default=[],
        help="Questions to run (can be an arbitary number of questions). Specify like so: q1 q2 q3"
    )

    return parser.parse_args()

# Function to find all matching files
def find_question_files(directory, pattern, questions_to_run):
    """
        questions_to_run: either will be an empty list (meaning we run all questions) or it will be a list of question prefixes like ["q1, "q2"]
    """
    question_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            # print(f"Checking file: {root}")
            if "ground_truth" in root:
                continue
            if re.match(pattern, file):
            # and "q1" not in file and "q2" not in file and "q3" not in file and "q4" not in file and "q5" not in file and "q6" not in file:  # Match the file pattern
                if questions_to_run: # if we specify the questions we want to run, then only run those questions.
                    if any(q in file for q in questions_to_run):
                        question_files.append(os.path.join(root, file))
                else:
                    question_files.append(os.path.join(root, file))
    return question_files

def get_log_file_openhands(question_file, unique_id, iteration, folder_prefix="llm_reasoning"):
    if not os.path.exists(f"../eval_metadata/temp_logs/openhands/{folder_prefix}"):
        os.makedirs(f"../eval_metadata/temp_logs/openhands/{folder_prefix}")
    log_filename =  f"../eval_metadata/temp_logs/openhands/{folder_prefix}/{os.path.basename(question_file).replace('.txt', '')}_{unique_id}_iter{iteration}.log"
    return os.path.abspath(log_filename) # abs filepath is fine since this refers to the host directory too. 

# def get_log_file_magentic(question_file, unique_id, iteration, folder_prefix="llm_reasoning"):
#     if not os.path.exists(f"../eval_metadata/temp_logs/magentic/{folder_prefix}"):
#         os.makedirs(f"../eval_metadata/temp_logs/magentic/{folder_prefix}")
#     log_filename =  f"../eval_metadata/temp_logs/magentic/{folder_prefix}/{os.path.basename(question_file).replace('.txt', '')}_{unique_id}_iter{iteration}.log"
#     return os.path.abspath(log_filename) # abs filepath is fine since this refers to the host directory too. 

# Function to create a configuration file
def create_config_file(question_file, unique_id, iteration, folder_prefix="llm_reasoning"):
    log_filename = f"/temp/logs/{os.path.basename(question_file).replace('.txt', '')}_{unique_id}_iter{iteration}.log"
    config_filename = f"../eval_metadata/configs/{folder_prefix}_config_{os.path.basename(question_file).replace('.txt', '')}_{unique_id}_iter{iteration}.json"

    if folder_prefix == "cloud_infra":
        config = {
            "benchmark_specific_context": "prompts/benchmark_specific/cloud-infra-questions-helper-context.txt", 
            "question_filename": question_file,
            "log_filename": log_filename,
            "supervisor_system_prompt_filename": "prompts/benchmark_specific/cloud-infra-supervisor.txt",
            "control_worker_system_prompt_filename": "prompts/benchmark_specific/cloud-infra-controlled-worker.txt",
        }
    else:
        config = {
            "benchmark_specific_context": "none",
            "question_filename": question_file,
            "log_filename": log_filename,
            "supervisor_system_prompt_filename": "prompts/benchmark_specific/llm-reasoning-supervisor.txt",
            "control_worker_system_prompt_filename": "prompts/benchmark_specific/llm-reasoning-controlled-worker.txt",
        }

    os.makedirs(os.path.dirname(config_filename), exist_ok=True)
    with open(config_filename, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Config file created: {config_filename}")
    return config_filename

# Function to run a Docker container
def run_docker_container(unique_id, iteration, folder_prefix="llm_reasoning"):
    rand_uuid = uuid.uuid4()
    container_name = f"exp-agent-container-{unique_id}-{rand_uuid}-iter{iteration}"
    print(f"Building Docker image for iteration {iteration}...")
    
    if folder_prefix == "llm_reasoning_2":
        image_name = "exp-agent-image-llm-reasoning-2"
        docker_filename = "ExpDockerfile_llm_reasoning_2"
    else:
        image_name = "exp-agent-image"
        docker_filename = "ExpDockerfile"

    subprocess.run(["sudo", "docker", "build", "-t", image_name, "-f", docker_filename, ".."], check=True)

    print(f"Running Docker container: {container_name}")
    host_dir = os.path.expanduser("~/langgraph-exp-agent/eval_metadata")
    subprocess.run([
        "sudo", "docker", "run", "--cpus=4", "--memory=8g", "--network=host", "-d",
        "--name", container_name,
        # "-v", f"{host_dir}:/eval_metadata", # remove since noticed that there is a small possibility that it may delete my logs folder on the host..
        image_name
    ], check=True)
    return container_name

# Function to execute the experiment inside the Docker container
def execute_experiment_in_container(container_name, config_file, folder_prefix="llm_reasoning"):
    """
    Executes the experiment inside the specified Docker container and retrieves log files.

    Args:
        container_name (str): The name of the Docker container.
        config_file (str): The path to the configuration file for the experiment.

    Raises:
        Exception: If any subprocess command fails.
    """
    print(f"Starting experiment in container {container_name} with config: {config_file}")
    try:
        # Run the experiment inside the container
        subprocess.run([
            "sudo", "docker", "exec", container_name,
            "bash", "-c", f"source ~/.bashrc && conda activate langgraph && python3 main.py {config_file}"
        ], check=True)  # This will block until main.py finishes
        print("Experiment completed successfully.")
        
        # Define source and destination directories
        container_log_dir = "/temp/logs/"  # Directory in the container
        host_log_dir = os.path.expanduser(f"../eval_metadata/temp_logs/{folder_prefix}")  # Host directory
        
        # Ensure host log directory exists
        os.makedirs(host_log_dir, exist_ok=True)
        
        # Run the find and tar command inside the container
        find_tar_cmd = [
            "sudo", "docker", "exec", container_name,
            "sh", "-c",
            f"cd {container_log_dir} && find . -maxdepth 1 -type f -name '*.log' | tar -cf - -T -"
        ]

        # Use Popen to handle piping
        with subprocess.Popen(find_tar_cmd, stdout=subprocess.PIPE) as proc1:
            subprocess.run(
                ["tar", "-xf", "-", "-C", host_log_dir],
                stdin=proc1.stdout,
                check=True
            )
            proc1.stdout.close()  # Close the stdout pipe after it's used
        
        print(f"Logs successfully extracted to {host_log_dir}")
    
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed with exit code {e.returncode}. Error: {e}")
        raise

# Function to stop and remove the Docker container
def cleanup_docker_container(container_name):
    print(f"Stopping and removing Docker container: {container_name}...")
    subprocess.run(["sudo", "docker", "stop", container_name], check=True)
    subprocess.run(["sudo", "docker", "rm", container_name], check=True)
    print(f"Docker container {container_name} cleaned up.")

def run_prune_commands():
    commands = [
        ["sudo", "docker", "container", "prune", "-f"],
        ["sudo", "docker", "image", "prune", "-f"],
        ["sudo", "docker", "volume", "prune", "-f"],
        ["sudo", "docker", "builder", "prune", "-f"],
    ]

    for command in commands:
        try:
            print(f"Running command: {' '.join(command)}")
            result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(result.stdout.decode())  # Print the standard output
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {' '.join(command)}")
            print(e.stderr.decode())  # Print the standard error

def execute_curie(question_file, unique_id, iteration, folder_prefix="llm_reasoning"):
    # Create configuration file
    config_file = create_config_file(question_file, unique_id, iteration, folder_prefix)

    # Run Docker container for this iteration
    container_name = None
    try:
        container_name = run_docker_container(unique_id, iteration, folder_prefix)

        # Execute experiment in Docker container
        execute_experiment_in_container(container_name, config_file, folder_prefix)

    finally:
        # Clean up Docker container after each iteration
        if container_name:
            cleanup_docker_container(container_name)
        run_prune_commands()
        # x=1

def cleanup_vdb_workspace():
    try:
        dirname = "../baselines/openhands/workspace/vector_index_related/starter_file"
        dirname = os.path.abspath(dirname)

        original_dirname = "../starter_file/faiss"
        original_dirname = os.path.abspath(original_dirname)
        # Combine all commands into one shell invocation
        subprocess.run(
            f"cd {dirname} && rm -rf faiss && cp -r {original_dirname} .",
            check=True,
            shell=True
        )
        print("Repository cloned successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

def cleanup_reasoning_workspace():
    try:
        dirname = "../baselines/openhands/workspace/llm_reasoning_related/starter_file"
        dirname = os.path.abspath(dirname)

        original_dirname = "../starter_file/large_language_monkeys"
        original_dirname = os.path.abspath(original_dirname)
        # Combine all commands into one shell invocation
        subprocess.run(
            f"cd {dirname} && rm -rf large_language_monkeys && cp -r {original_dirname} .",
            check=True,
            shell=True
        )
        print("Repository cloned successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

def cleanup_reasoning2_workspace():
    try:
        dirname = "../baselines/openhands/workspace/llm_reasoning_2_related/starter_file"
        dirname = os.path.abspath(dirname)

        original_dirname = "../starter_file/The-Impact-of-Reasoning-Step-Length-on-Large-Language-Models"
        original_dirname = os.path.abspath(original_dirname)
        # Combine all commands into one shell invocation
        subprocess.run(
            f"cd {dirname} && rm -rf The-Impact-of-Reasoning-Step-Length-on-Large-Language-Models && cp -r {original_dirname} .",
            check=True,
            shell=True
        )
        print("Repository cloned successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

def cleanup_ml_training_workspace():
    try:
        dirname = "../baselines/openhands/workspace/ml_training_related/starter_file"
        dirname = os.path.abspath(dirname)

        original_dirname = "../starter_file/MLAgentBench"
        original_dirname = os.path.abspath(original_dirname)
        # Combine all commands into one shell invocation
        subprocess.run(
            f"cd {dirname} && rm -rf MLAgentBench && cp -r {original_dirname} .",
            check=True,
            shell=True
        )
        print("Repository cloned successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

def cleanup_cloud_workspace():
    try:
        dirname = "../baselines/openhands/workspace/cloud_infra_related/starter_file"
        dirname = os.path.abspath(dirname)

        original_dirname = "../starter_file/cloud_infra"
        original_dirname = os.path.abspath(original_dirname)
        # Combine all commands into one shell invocation
        subprocess.run(
            f"cd {dirname} && rm -rf cloud_infra && cp -r {original_dirname} .",
            check=True,
            shell=True
        )
        print("Repository cloned successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

def execute_openhands(question_file, unique_id, iteration, folder_prefix="llm_reasoning"):
    # question_file is an absolute path filename

    # Cleanup workspace:
    if folder_prefix == "llm_reasoning":
        cleanup_reasoning_workspace()
    elif folder_prefix == "vector_index":
        cleanup_vdb_workspace()
    elif folder_prefix == "cloud_infra":
        cleanup_cloud_workspace()
    elif folder_prefix == "ml_training":
        cleanup_ml_training_workspace()
    elif folder_prefix == "llm_reasoning_2":
        cleanup_reasoning2_workspace()

    # Get log file for openhands:
    log_filename = get_log_file_openhands(question_file, unique_id, iteration, folder_prefix)
    print(f"Log file for openhands: {log_filename}")
    openhands_dir = "~/OpenHands"
    openhands_dir = os.path.expanduser(openhands_dir)

    common_txt_filename = f"~/langgraph-exp-agent/benchmark/common.txt"
    common_txt_filename = os.path.expanduser(common_txt_filename)

    # Read from common_txt_filename:
    question_text = ""
    with open(common_txt_filename, 'r') as f:
        question_text = f.read()
    # Read from question_file: but replace all directories with the correct directory. LLM reasoning qs now are specified in terms of the docker container filepaths, we need to convert them into host paths for openhands, and they also need to be in the workspace. 
    with open(question_file, 'r') as f:
        q_temp = f.read()
        # starter_file_dir = "~/langgraph-exp-agent/starter_file"
        # starter_file_dir = os.path.expanduser(starter_file_dir)
        starter_file_dir = f"{folder_prefix}_related/starter_file"
        # Convert all occurences of "/starter_file" to starter_file_dir (which may be something like llm_reasoning_related/starter_file) in the question file:
        q_temp = q_temp.replace("/starter_file", starter_file_dir)

        if folder_prefix == "llm_reasoning_2":
            with open("setup/env.sh", 'r') as f:
                env_credentials = f.read()

            # Convert all occurences of the following:
            q_temp = q_temp.replace("source /exp_agent/setup/env.sh", env_credentials)
            q_temp = q_temp.replace("conda activate impact", f"conda activate impact; pip install torch==1.8.2+cu111 torchtext==0.9.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html; pip install -r {starter_file_dir}/The-Impact-of-Reasoning-Step-Length-on-Large-Language-Models/requirements.txt")

        question_text += q_temp
    # Save to a new file:
    updated_ques_filename = f"~/langgraph-exp-agent/eval_metadata/temp_scripts/openhands/{os.path.basename(question_file)}"
    updated_ques_filename = os.path.expanduser(updated_ques_filename)
    with open(updated_ques_filename, 'w') as f:
        f.write(question_text)

    # Construct the command
    command = f"""
export LOG_ALL_EVENTS=true
cd {openhands_dir}
poetry run python -m openhands.core.main -f {updated_ques_filename} 2>&1 | tee -a {log_filename}
    """

    # Execute the command
    try:
        print(f"Executing OpenHands pipeline for question {question_file}, iteration {iteration}...")
        subprocess.run(command, shell=True, check=True, executable="/bin/bash")
        print(f"Completed iteration {iteration} for question {question_file}. Logs stored in {log_filename}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing pipeline for {question_file}, iteration {iteration}: {e}")
        raise

def run_magentic_docker_container(unique_id, iteration, PROMPT_FILE, folder_prefix="llm_reasoning"):
    rand_uuid = uuid.uuid4()
    container_name = f"magentic-agent-container-{unique_id}-{rand_uuid}-iter{iteration}"
    print(f"Building Docker image for iteration {iteration}...")
    
    image_name = "magentic-agent-image"
    docker_filename = "MagenticDockerfile"

    command = [
        "sudo", "docker", "build",
        "--build-arg", f"PROMPT_FILE={PROMPT_FILE}",
        "-t", "magentic-agent-image",
        "-f", "MagenticDockerfile",
        ".."
    ]

    subprocess.run(command, check=True)

    print(f"Running Docker container: {container_name}")
    command = [
        "sudo", "docker", "run",
        "-v", "/var/run/docker.sock:/var/run/docker.sock",
        "-v", "/usr/bin/docker:/usr/bin/docker",
        "--cpus=4",
        "--memory=8g",
        "--network=host",
        "-d",
        "--name", container_name,
        image_name
    ]

    subprocess.run(command, check=True)
    return container_name

def execute_experiment_in_magentic_container(container_name, question_file, folder_prefix="llm_reasoning"):
    """
    Executes the experiment inside the specified Docker container and retrieves log files.

    Args:
        container_name (str): The name of the Docker container.
        config_file (str): The path to the configuration file for the experiment.

    Raises:
        Exception: If any subprocess command fails.
    """
    print(f"Starting experiment in container {container_name} with question_file: {question_file}")
    try:
        # Run the experiment inside the container
        subprocess.run([
            "sudo", "docker", "exec", container_name,
            "bash", "-c", f"source ~/.bashrc && python /starter_file/autogen/python/packages/autogen-magentic-one/examples/example.py  --logs_dir /temp/logs"
        ], check=True)  # This will block until main.py finishes
        print("Experiment completed successfully.")
        
        # Define source and destination directories
        container_log_dir = "/temp/logs/"  # Directory in the container
        host_log_dir = os.path.expanduser(f"../eval_metadata/temp_logs/magentic/{folder_prefix}")  # Host directory
        
        # Ensure host log directory exists
        os.makedirs(host_log_dir, exist_ok=True)
        
        # Run the find and tar command inside the container
        find_tar_cmd = [
            "sudo", "docker", "exec", container_name,
            "sh", "-c",
            f"cd {container_log_dir} && find . -maxdepth 1 -type f -name '*.jsonl' | tar -cf - -T -"
        ]

        log_suffix = os.path.basename(question_file).replace('.txt', '')

        rename_and_tar_cmd = [
            "sudo", "docker", "exec", container_name,
            "sh", "-c",
            f"""
            cd {container_log_dir} && \
            find . -maxdepth 1 -type f -name '*.jsonl' -exec sh -c '
                for file; do
                    mv "$file" "${{file%.jsonl}}_{log_suffix}.jsonl";
                done
            ' _ {{}} + && \
            find . -maxdepth 1 -type f -name '*.jsonl' | tar -cf - -T -
            """
        ]

        # Use Popen to handle piping
        with subprocess.Popen(rename_and_tar_cmd, stdout=subprocess.PIPE) as proc1:
            subprocess.run(
                ["tar", "-xf", "-", "-C", host_log_dir],
                stdin=proc1.stdout,
                check=True
            )
            proc1.stdout.close()  # Close the stdout pipe after it's used
        
        print(f"Logs successfully extracted to {host_log_dir}")
    
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed with exit code {e.returncode}. Error: {e}")
        raise

def execute_magentic(question_file, unique_id, iteration, folder_prefix="llm_reasoning"):
    # Get log file for openhands:

    common_txt_filename = f"~/langgraph-exp-agent/benchmark/common.txt"
    common_txt_filename = os.path.expanduser(common_txt_filename)

    # Read from common_txt_filename:
    question_text = ""
    with open(common_txt_filename, 'r') as f:
        question_text = f.read()
    # Read from question_file: but replace all directories with the correct directory. LLM reasoning qs now are specified in terms of the docker container filepaths, we need to convert them into host paths for openhands, and they also need to be in the workspace. 
    with open(question_file, 'r') as f:
        q_temp = f.read()

        if folder_prefix == "llm_reasoning_2":
            # Convert all occurences of the following:
            q_temp = q_temp.replace("conda activate impact", f"conda activate impact; pip install torch==1.8.2+cu111 torchtext==0.9.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html; pip install -r /starter_file/The-Impact-of-Reasoning-Step-Length-on-Large-Language-Models/requirements.txt")

        question_text += q_temp
    # Save to a new file:
    updated_ques_filename = f"../eval_metadata/temp_scripts/magentic/{os.path.basename(question_file)}"
    if not os.path.exists("../eval_metadata/temp_scripts/magentic/"):
        os.makedirs("../eval_metadata/temp_scripts/magentic/")
    updated_ques_filename = os.path.expanduser(updated_ques_filename)
    with open(updated_ques_filename, 'w') as f:
        f.write(question_text)
    
    PROMPT_FILE = f"/eval_metadata/temp_scripts/magentic/{os.path.basename(question_file)}"

    # Run Docker container for this iteration
    container_name = None
    try:
        container_name = run_magentic_docker_container(unique_id, iteration, PROMPT_FILE, folder_prefix)

        # Execute experiment in Docker container
        execute_experiment_in_magentic_container(container_name, question_file, folder_prefix)

    finally:
        # Clean up Docker container after each iteration
        if container_name:
            cleanup_docker_container(container_name)
        run_prune_commands()
        # x=1

# Main function
def main():
    args = parse_args()
    
    print(f"Iterations: {args.iterations}")
    print(f"Pipeline: {args.pipeline}")
    print(f"Timeout: {args.timeout} seconds")

    if args.category == QCategory.REASONING:
        folder_prefix = "llm_reasoning"
    elif args.category == QCategory.VDB:
        folder_prefix = "vector_index"
    elif args.category == QCategory.CLOUD:
        folder_prefix = "cloud_infra"
    elif args.category == QCategory.MLTRAINING:
        folder_prefix = "ml_training"
    elif args.category == QCategory.REASONING2:
        folder_prefix = "llm_reasoning_2"

    # Directory and file pattern
    questions_dir = f"../benchmark/{folder_prefix}"
    file_pattern = r"^q.*\.txt$"  # Pattern for q<string>.txt

    # Find all matching question files
    question_files = find_question_files(questions_dir, file_pattern, args.questions_to_run)
    print(f"Found {len(question_files)} question files:")
    for question_file in question_files:
        print(f"  - {question_file}")

    # Iterate through each question file
    for question_file in question_files:
        print(f"Processing {question_file} for {args.iterations} iterations...")
        for iteration in range(1, args.iterations + 1):
            start_time = time.time()
            # Perform the required operation for each iteration (to be provided later)
            print(f"Iteration {iteration} for {question_file}...")

            # Unique identifier for log and config filenames
            unique_id = datetime.now().strftime("%Y%m%d%H%M%S")

            if args.pipeline == Pipeline.OPENHANDS:
                question_file = os.path.abspath(question_file) # question_file is a relative path in this format: ../benchmark/llm_reasoning/q4_target_coverage.txt. We convert it to an abs path for openhands since we need to pass in the abs dir. 
                execute_openhands(question_file, unique_id, iteration, folder_prefix)
            elif args.pipeline == Pipeline.CURIE:
                execute_curie(question_file, unique_id, iteration, folder_prefix)
            elif args.pipeline == Pipeline.MAGENTIC:
                execute_magentic(question_file, unique_id, iteration, folder_prefix)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Iteration {iteration} for {question_file} completed in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    # Create a main loop log file based on a uuid:
    uuid_time = datetime.now().strftime("%Y%m%d%H%M%S")
    main_loop_log_filename = f"../eval_metadata/logs/main_loop/main_loop_logs_{uuid_time}.log" # logs main loop metadata
    main_loop_log_file = open(main_loop_log_filename, 'w')
    sys.stdout = main_loop_log_file
    sys.stderr = main_loop_log_file

    main()

    # Reset stdout and stderr
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    print(f"Output logged to {main_loop_log_filename}")