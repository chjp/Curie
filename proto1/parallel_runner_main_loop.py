# Currently, we parallelize based on the questions. Each question may run for multiple iterations but that is within a single process.
import subprocess
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

def update_makefile(makefile_path, updates):
    """
    Updates variables in a Makefile dynamically.

    :param makefile_path: Path to the Makefile.
    :param updates: Dictionary of variable names and their new values.
    """
    with open(makefile_path, 'r') as file:
        lines = file.readlines()

    # Regex to match variable assignments
    variable_pattern = re.compile(r"^(\w+)\s*=\s*(.+)$")

    updated_lines = []
    for line in lines:
        match = variable_pattern.match(line)
        if match:
            var_name, current_value = match.groups()
            # Update the variable if it's in the updates dictionary
            if var_name in updates:
                new_value = updates[var_name]
                updated_line = f"{var_name} = {new_value}\n"
                updated_lines.append(updated_line)
            else:
                updated_lines.append(line)
        else:
            updated_lines.append(line)

    # Write the updated content back to the Makefile
    with open(makefile_path, 'w') as file:
        file.writelines(updated_lines)

def run_command(task, questions, iter_now=0):
    """
    Run the Python command for a group of questions.
    """
    print("Running command for task: {}, questions: {}".format(task, questions))
    questions_to_run = " ".join(questions)  # Combine questions into a single string
    command = [
        "python3", "main_loop.py",
        "--iterations", str(task["iterations"]),
        "--pipeline", task["pipeline"],
        "--timeout", str(task["timeout"]),
        "--category", task["category"],
        "--questions_to_run", questions_to_run
    ]
    try:
        if task["pipeline"] == "openhands":
            # Update makefile to use new ports for each parallel execution:
            makefile_path = "/home/patkon/OpenHands/Makefile"
            updates = {
                "BACKEND_PORT": str(3000+iter_now),         # Change backend port
                "FRONTEND_PORT": str(3001+iter_now)        # Change frontend port
            }

            update_makefile(makefile_path, updates)

        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Command completed for {task} with questions '{questions_to_run}': {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Command failed for {task} with questions '{questions_to_run}': {e}")

def main():
    parser = argparse.ArgumentParser(description="Run multiple configurations in parallel.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to a JSON file containing the configuration array.")
    parser.add_argument("--parallel_runs", type=int, default=25,
                        help="Number of parallel processes to run (default is 1).")
    parser.add_argument("--questions_per_submit", type=int, default=1,
                        help="Number of questions to include in a single command submission.")
    args = parser.parse_args()

    # Load the JSON configuration file
    with open(args.config, "r") as config_file:
        tasks = json.load(config_file)

    # Run tasks in parallel
    with ThreadPoolExecutor(max_workers=args.parallel_runs) as executor:
        future_to_task = {}

        # Iterate over tasks and break down questions_to_run
        iter_now = 0
        for task in tasks:
            questions = task["questions_to_run"].split()  # Split questions into individual strings
            # Group questions based on the --questions_per_submit value
            grouped_questions = [
                questions[i:i + args.questions_per_submit]
                for i in range(0, len(questions), args.questions_per_submit)
            ]
            for group in grouped_questions:
                # Submit each group of questions as a single task
                future = executor.submit(run_command, task, group, iter_now)
                future_to_task[future] = {"task": task, "questions": group}
                iter_now += 1
                time.sleep(180) # Sleep to allow time to create docker files and initiate the run

        # Wait for all futures to complete
        for future in as_completed(future_to_task):
            task_info = future_to_task[future]
            try:
                future.result()  # This will raise any exceptions that occurred during execution
            except Exception as e:
                print(f"Error with task {task_info}: {e}")

if __name__ == "__main__":
    main()