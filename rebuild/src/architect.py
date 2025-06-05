"""
Architect Module - Handles experiment planning

This module implements a simplified version of the Architect agent
that plans experiments based on research questions.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger("curie")

class Architect:
    """Architect agent for planning experiments"""
    
    def __init__(self, prompt_template: Optional[str] = None):
        """Initialize the architect
        
        Args:
            prompt_template: Path to prompt template file for the architect
        """
        self.prompt_template = prompt_template
        
    def load_prompt(self) -> str:
        """Load the architect prompt template"""
        if not self.prompt_template or not os.path.exists(self.prompt_template):
            # Use default prompt if template file doesn't exist
            return """
            As a scientific experiment architect, plan an experiment to answer this question: {{question}}
            
            Your plan should include the following steps:
            1. Literature review
            2. Hypothesis formulation
            3. Experimental design
            4. Data collection
            5. Analysis and conclusions
            
            For each step, provide a clear description of what should be done.
            """
            
        with open(self.prompt_template, 'r') as f:
            return f.read()
            
    def prepare_prompt(self, question: str) -> str:
        """Prepare the architect prompt for the specific question"""
        template = self.load_prompt()
        prompt = template.replace("{{question}}", question)
        return prompt
        
    def plan_experiment(self, question: str, dataset_dir: Optional[str] = None) -> Dict[str, Any]:
        """Generate an experiment plan for the given research question
        
        Args:
            question: The research question to plan an experiment for
            dataset_dir: Optional path to dataset directory
            
        Returns:
            Dictionary containing the experiment plan
        """
        logger.info(f"Architect planning experiment for question: {question}")
        
        # Prepare prompt
        prompt = self.prepare_prompt(question)
        
        # In a real implementation, this would call the LLM to generate a plan
        # For MVP, we'll create a simple predefined plan
        
        # Examine the question to customize the plan slightly
        if "machine learning" in question.lower() or "ml" in question.lower():
            return self._create_ml_plan(question)
        elif "data" in question.lower() or "analysis" in question.lower():
            return self._create_data_analysis_plan(question)
        else:
            return self._create_general_plan(question)
    
    def _create_general_plan(self, question: str) -> Dict[str, Any]:
        """Create a general experiment plan"""
        return {
            "title": f"Research Plan for: {question}",
            "question": question,
            "steps": [
                {
                    "name": "literature_review",
                    "description": "Review existing knowledge on the topic to identify relevant theories, methods, and gaps",
                    "expected_output": "Summary of key findings from literature"
                },
                {
                    "name": "hypothesis",
                    "description": "Formulate a testable hypothesis based on the literature review",
                    "expected_output": "Primary and null hypotheses"
                },
                {
                    "name": "experiment_design",
                    "description": "Design a controlled experiment to test the hypothesis, including variables, controls, and methodology",
                    "expected_output": "Detailed experiment protocol"
                },
                {
                    "name": "data_collection",
                    "description": "Collect data according to the experiment design",
                    "expected_output": "Raw dataset ready for analysis"
                },
                {
                    "name": "analysis",
                    "description": "Analyze the data using appropriate statistical methods and draw conclusions",
                    "expected_output": "Statistical results and interpretation"
                }
            ]
        }
    
    def _create_ml_plan(self, question: str) -> Dict[str, Any]:
        """Create a plan focused on machine learning experiments"""
        return {
            "title": f"Machine Learning Research Plan for: {question}",
            "question": question,
            "steps": [
                {
                    "name": "literature_review",
                    "description": "Review existing ML approaches related to the question, focusing on algorithms, architectures, and evaluation metrics",
                    "expected_output": "Summary of relevant ML techniques and benchmarks"
                },
                {
                    "name": "hypothesis",
                    "description": "Formulate hypotheses about which ML approaches will perform best for the given problem",
                    "expected_output": "Testable hypotheses comparing different ML approaches"
                },
                {
                    "name": "experiment_design",
                    "description": "Design experiments to compare ML models, including data preprocessing, model selection, hyperparameter tuning, and evaluation protocols",
                    "expected_output": "ML experiment workflow and evaluation criteria"
                },
                {
                    "name": "data_collection",
                    "description": "Prepare datasets for training and evaluation, including data cleaning, feature engineering, and splitting into train/validation/test sets",
                    "expected_output": "Processed datasets ready for model training"
                },
                {
                    "name": "model_development",
                    "description": "Implement and train the ML models according to the experiment design",
                    "expected_output": "Trained models and training metrics"
                },
                {
                    "name": "analysis",
                    "description": "Evaluate models on test data, compare performance, and interpret results",
                    "expected_output": "Model comparisons and insights with visualizations"
                }
            ]
        }
    
    def _create_data_analysis_plan(self, question: str) -> Dict[str, Any]:
        """Create a plan focused on data analysis"""
        return {
            "title": f"Data Analysis Plan for: {question}",
            "question": question,
            "steps": [
                {
                    "name": "literature_review",
                    "description": "Review existing analytical approaches for similar data and questions",
                    "expected_output": "Summary of relevant analytical methods"
                },
                {
                    "name": "hypothesis",
                    "description": "Formulate hypotheses about patterns or relationships in the data",
                    "expected_output": "Testable hypotheses about data patterns"
                },
                {
                    "name": "data_preparation",
                    "description": "Clean, preprocess, and explore the dataset to understand its properties",
                    "expected_output": "Processed dataset and exploratory analysis"
                },
                {
                    "name": "analysis_design",
                    "description": "Design analytical approach including statistical tests or models to apply",
                    "expected_output": "Analytical workflow plan"
                },
                {
                    "name": "analysis",
                    "description": "Apply statistical methods, create visualizations, and interpret results",
                    "expected_output": "Statistical results, visualizations, and interpretations"
                },
                {
                    "name": "conclusions",
                    "description": "Draw conclusions based on the analysis and address the original research question",
                    "expected_output": "Final conclusions and insights"
                }
            ]
        }
    
