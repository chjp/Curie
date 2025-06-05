"""
Main Module - Entry point for Curie MVP

This module provides the main interfaces for running Curie experiments.
"""

import os
import json
import logging
from typing import Dict, Any, Optional

from .experiment import Experiment, run_experiment
from .architect import Architect
from .worker import Worker
from .validator import Validator, PatcherValidator
from .reporter import Reporter

def init_logger(log_file: Optional[str] = None) -> logging.Logger:
    """Initialize and configure logger"""
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
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

def curie(
    question: str,
    workspace_dir: Optional[str] = None,
    dataset_dir: Optional[str] = None,
    api_keys: Optional[Dict[str, str]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Main entry point for running Curie experiments
    
    Args:
        question: The research question to investigate
        workspace_dir: Directory to store experiment files and code
        dataset_dir: Directory containing datasets for the experiment
        api_keys: Dictionary of API keys for LLM services
        config: Additional configuration parameters
        
    Returns:
        Dictionary containing experiment results
    """
    # Set default config values
    if config is None:
        config = {}
    
    default_config = {
        "max_steps": 5,
        "model": "gpt-4",
        "prompts_dir": os.path.join(os.path.dirname(__file__), "..", "prompts"),
        "log_level": "INFO"
    }
    
    for key, value in default_config.items():
        if key not in config:
            config[key] = value
    
    # Setup workspace and output directories
    base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    workspace_dir = workspace_dir or os.path.join(base_dir, "workspace")
    output_dir = os.path.join(base_dir, "output")
    
    os.makedirs(workspace_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize logger
    log_file = os.path.join(output_dir, "curie.log")
    logger = init_logger(log_file)
    logger.info(f"Starting Curie experiment for question: {question}")
    
    # Run the experiment
    result = run_experiment(
        question=question,
        workspace_dir=workspace_dir,
        dataset_dir=dataset_dir,
        api_keys=api_keys,
        config=config
    )
    
    logger.info("Experiment complete")
    return result

if __name__ == "__main__":
    # Simple example of running the MVP directly
    results = curie(
        question="How does choice of sorting algorithm impact runtime performance across different input distributions?",
        config={"max_steps": 5}
    )
    
    print(f"Experiment completed with status: {results.get('status', 'unknown')}")
    print(f"Check output directory for the report.")
