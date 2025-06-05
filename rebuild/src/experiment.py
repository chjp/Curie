"""
Experiment Module - Main entry point for Curie MVP

This module provides the core functionality for running scientific experiments
in a simplified version of the Curie framework.
"""

import os
import json
import time
import uuid
from datetime import datetime
import logging
from typing import Dict, Optional, Any

# Set up logging
def setup_logger(log_file: str = None) -> logging.Logger:
    """Set up and configure logger for the experiment"""
    logger = logging.getLogger("curie")
    logger.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if log file is provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

class Experiment:
    """Core experiment class for running scientific experiments"""
    
    def __init__(
        self,
        question: str,
        workspace_dir: str = None,
        dataset_dir: str = None,
        api_keys: Dict[str, str] = None,
        config: Dict[str, Any] = None
    ):
        """Initialize the experiment with configurations
        
        Args:
            question: The research question to investigate
            workspace_dir: Directory to store experiment files and code
            dataset_dir: Directory containing datasets for the experiment
            api_keys: Dictionary of API keys for LLM services
            config: Additional configuration parameters
        """
        self.question = question
        self.experiment_id = f"{int(time.time())}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Setup directories
        self.workspace_dir = workspace_dir or os.path.join(os.getcwd(), "workspace")
        self.dataset_dir = dataset_dir
        self.output_dir = os.path.join(os.getcwd(), "logs")
        
        # Ensure directories exist
        os.makedirs(self.workspace_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup config
        self.config = self._setup_config(config)
        
        # Setup API keys
        self.api_keys = api_keys or {}
        
        # Setup logger
        log_file = os.path.join(self.output_dir, f"experiment_{self.experiment_id}.log")
        self.logger = setup_logger(log_file)
        
        # Store experiment state
        self.state = {
            "status": "initialized",
            "steps": [],
            "results": None,
        }
    
    def _setup_config(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Set up experiment configuration with defaults"""
        default_config = {
            "max_steps": 5,
            "model": "gpt-4",
            "prompts_dir": os.path.join(os.path.dirname(__file__), "..", "prompts"),
        }
        
        if config:
            default_config.update(config)
        
        return default_config
    
    def _save_question(self):
        """Save the research question to a file"""
        question_file = os.path.join(self.workspace_dir, f"question_{self.experiment_id}.txt")
        with open(question_file, 'w') as f:
            f.write(self.question)
        return question_file
    
    def _save_experiment_state(self):
        """Save the current experiment state to a file"""
        state_file = os.path.join(self.output_dir, f"state_{self.experiment_id}.json")
        with open(state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def plan_experiment(self):
        """Plan the experiment based on the research question"""
        self.logger.info("Planning experiment for question: %s", self.question)
        self.state["status"] = "planning"
        
        # In a real implementation, this would call the Architect agent to plan the experiment
        # For the MVP, we'll just add a simple placeholder plan
        
        plan = {
            "title": f"Research on: {self.question}",
            "steps": [
                {"name": "literature_review", "description": "Review existing knowledge on the topic"},
                {"name": "hypothesis", "description": "Formulate a testable hypothesis"},
                {"name": "experiment_design", "description": "Design experiment to test hypothesis"},
                {"name": "data_collection", "description": "Collect or generate necessary data"},
                {"name": "analysis", "description": "Analyze the results"}
            ]
        }
        
        self.state["plan"] = plan
        self._save_experiment_state()
        
        return plan
    
    def execute_step(self, step_name):
        """Execute a single step of the experiment"""
        if "plan" not in self.state:
            self.plan_experiment()
            
        step_found = False
        for step in self.state["plan"]["steps"]:
            if step["name"] == step_name:
                step_found = True
                break
                
        if not step_found:
            self.logger.error(f"Step {step_name} not found in experiment plan")
            return False
            
        self.logger.info(f"Executing step: {step_name}")
        self.state["status"] = f"executing_{step_name}"
        
        # Here we would invoke the worker agent to execute the step
        # For MVP, we'll just log that the step is being executed
        
        # Add result to state
        step_result = {
            "step": step_name,
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "output": f"Example output for {step_name}"
        }
        
        self.state["steps"].append(step_result)
        self._save_experiment_state()
        
        return step_result
    
    def execute_experiment(self):
        """Execute the full experiment based on the plan"""
        self.logger.info("Starting experiment execution")
        self._save_question()
        
        # Get or create plan
        if "plan" not in self.state:
            plan = self.plan_experiment()
        else:
            plan = self.state["plan"]
            
        # Execute each step in the plan
        for step in plan["steps"]:
            self.execute_step(step["name"])
            
        # Generate report
        self.generate_report()
        
        self.state["status"] = "completed"
        self._save_experiment_state()
        
        self.logger.info("Experiment execution completed")
        return self.state
    
    def generate_report(self):
        """Generate a report for the experiment"""
        self.logger.info("Generating experiment report")
        
        # In a real implementation, this would use the reporter/concluder agent
        # For MVP, we'll create a simple report
        
        report = {
            "title": f"Research Report: {self.question}",
            "experiment_id": self.experiment_id,
            "timestamp": datetime.now().isoformat(),
            "steps": self.state["steps"],
            "conclusion": "This is a placeholder conclusion for the MVP."
        }
        
        # Save report to file
        report_file = os.path.join(self.output_dir, f"report_{self.experiment_id}.md")
        with open(report_file, 'w') as f:
            f.write(f"# {report['title']}\n\n")
            f.write(f"**Experiment ID**: {report['experiment_id']}\n")
            f.write(f"**Date**: {report['timestamp']}\n\n")
            
            f.write("## Question\n\n")
            f.write(f"{self.question}\n\n")
            
            f.write("## Steps\n\n")
            for step in report['steps']:
                f.write(f"### {step['step'].replace('_', ' ').title()}\n\n")
                f.write(f"**Status**: {step['status']}\n")
                f.write(f"**Timestamp**: {step['timestamp']}\n\n")
                f.write(f"{step['output']}\n\n")
                
            f.write("## Conclusion\n\n")
            f.write(f"{report['conclusion']}\n")
        
        self.state["report_file"] = report_file
        return report_file


def run_experiment(
    question: str,
    workspace_dir: str = None,
    dataset_dir: str = None,
    api_keys: Dict[str, str] = None,
    config: Dict[str, Any] = None
) -> Dict:
    """Run a complete scientific experiment using the Curie framework
    
    Args:
        question: The research question to investigate
        workspace_dir: Directory to store experiment files and code
        dataset_dir: Directory containing datasets for the experiment
        api_keys: Dictionary of API keys for LLM services
        config: Additional configuration parameters
        
    Returns:
        Dictionary containing experiment state and results
    """
    experiment = Experiment(
        question=question,
        workspace_dir=workspace_dir,
        dataset_dir=dataset_dir,
        api_keys=api_keys,
        config=config
    )
    
    return experiment.execute_experiment()
