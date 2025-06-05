"""
Worker Module - Handles execution of experiment steps

This module implements a simplified version of the Worker agent 
that executes individual experiment steps.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger("curie")

class Worker:
    """Worker agent for executing experiment steps"""
    
    def __init__(self, workspace_dir: str, prompt_template: Optional[str] = None):
        """Initialize the worker
        
        Args:
            workspace_dir: Directory where experiment files and code will be stored
            prompt_template: Path to prompt template file for the worker
        """
        self.workspace_dir = workspace_dir
        self.prompt_template = prompt_template
        self.execution_history = []
        
    def load_prompt(self) -> str:
        """Load the worker prompt template"""
        if not self.prompt_template or not os.path.exists(self.prompt_template):
            # Use default prompt if template file doesn't exist
            return """
            As a scientific experiment worker, execute this step: {{step_name}}
            
            Research Question: {{question}}
            
            Step Description: {{step_description}}
            
            Execute this step with scientific rigor and attention to detail.
            """
            
        with open(self.prompt_template, 'r') as f:
            return f.read()
            
    def prepare_prompt(self, question: str, step_name: str, step_description: str) -> str:
        """Prepare the worker prompt for the specific step"""
        template = self.load_prompt()
        
        # Simple template replacement
        prompt = template.replace("{{question}}", question)
        prompt = prompt.replace("{{step_name}}", step_name)
        prompt = prompt.replace("{{step_description}}", step_description)
        
        return prompt
        
    def execute_code(self, code: str) -> Tuple[bool, str]:
        """Execute code in the workspace environment
        
        In the full implementation, this would safely execute code in a Docker container.
        For MVP, we'll just simulate execution.
        """
        logger.info(f"Simulating code execution: {code[:100]}...")
        
        # In a real implementation, this would execute code in a safe environment
        # For MVP, we just pretend it executed successfully
        
        return True, "Code executed successfully (simulated)"
        
    def execute_step(self, question: str, step_name: str, step_description: str) -> Dict[str, Any]:
        """Execute a single experiment step
        
        Args:
            question: The research question being investigated
            step_name: Name of the step to execute
            step_description: Description of the step to execute
            
        Returns:
            Dictionary containing step results
        """
        logger.info(f"Worker executing step: {step_name}")
        
        # Prepare prompt for the step
        prompt = self.prepare_prompt(question, step_name, step_description)
        
        # In a real implementation, this would call the LLM to generate a response
        # For MVP, we'll simulate the LLM response with a simple step execution
        
        # Simulate different responses based on step name
        if step_name == "literature_review":
            content = self._simulate_literature_review(question)
        elif step_name == "hypothesis":
            content = self._simulate_hypothesis(question)
        elif step_name == "experiment_design":
            content = self._simulate_experiment_design(question)
        elif step_name == "data_collection":
            content = self._simulate_data_collection(question)
        elif step_name == "analysis":
            content = self._simulate_analysis(question)
        else:
            content = f"Simulated step execution for {step_name}"
            
        # Save step output to workspace
        output_file = os.path.join(self.workspace_dir, f"{step_name}_output.txt")
        with open(output_file, 'w') as f:
            f.write(content)
            
        # Record in history
        step_result = {
            "step": step_name,
            "description": step_description,
            "output": content,
            "output_file": output_file,
            "status": "completed"
        }
        
        self.execution_history.append(step_result)
        
        return step_result
        
    def _simulate_literature_review(self, question: str) -> str:
        """Simulate a literature review response"""
        return f"""# Literature Review: {question}

Based on a thorough review of relevant literature, several key findings emerge:

1. Previous research has explored similar questions with mixed results
2. Smith et al. (2024) conducted experiments that showed promising outcomes
3. The theoretical framework proposed by Johnson (2023) provides a foundation for our investigation
4. There are gaps in existing literature regarding specific conditions and variables

This literature review suggests our research question is both novel and significant.
"""
        
    def _simulate_hypothesis(self, question: str) -> str:
        """Simulate hypothesis formulation"""
        return f"""# Hypothesis Formulation

Based on the literature review and the research question: "{question}"

## Primary Hypothesis
If variable A is manipulated under controlled conditions, then outcome B will increase significantly compared to the control.

## Null Hypothesis
There is no significant difference in outcome B when variable A is manipulated under controlled conditions compared to the control.

## Justification
This hypothesis is testable, specific, and aligns with previous theoretical frameworks while addressing the gaps identified in the literature review.
"""
        
    def _simulate_experiment_design(self, question: str) -> str:
        """Simulate experiment design"""
        return f"""# Experiment Design

## Research Question
"{question}"

## Variables
- Independent Variable: A with levels (A1, A2, A3)
- Dependent Variable: B (measured in units of X)
- Control Variables: C, D, E (held constant)

## Methodology
1. Random assignment of subjects to conditions
2. Double-blind procedure to eliminate bias
3. Pre-test and post-test measurements
4. Statistical analysis using ANOVA and post-hoc tests

## Sample Size
Based on power analysis with α=0.05 and β=0.2, the required sample size is 30 per condition.

## Controls
Appropriate controls include a placebo group and standard baseline measurements.
"""
        
    def _simulate_data_collection(self, question: str) -> str:
        """Simulate data collection"""
        return f"""# Data Collection Results

## Summary Statistics
- Total samples collected: 120
- Condition A1: n=40, mean=24.3, SD=3.2
- Condition A2: n=40, mean=29.7, SD=2.8
- Condition A3: n=40, mean=18.5, SD=4.1

## Quality Control
- 3 outliers identified and removed based on Grubbs test
- Data normality confirmed via Shapiro-Wilk test (p > 0.05)
- No missing values in the final dataset

## Raw Data Sample
```
Sample_ID,Condition,Value_B,Control_C,Control_D,Control_E
001,A1,23.1,5.0,TRUE,normal
002,A1,25.4,5.0,TRUE,normal
003,A1,22.8,5.0,TRUE,normal
...
118,A3,17.2,5.0,TRUE,normal
119,A3,19.5,5.0,TRUE,normal
120,A3,16.8,5.0,TRUE,normal
```
"""
        
    def _simulate_analysis(self, question: str) -> str:
        """Simulate analysis results"""
        return f"""# Analysis Results

## Statistical Tests
- One-way ANOVA: F(2,117) = 28.64, p < 0.001
- Post-hoc Tukey HSD: Condition A2 significantly higher than A1 (p=0.003) and A3 (p<0.001)

## Visualization
```
Boxplot showing distribution across conditions:
A1: [==|===]
A2: [====|=====]
A3: [==|==]
```

## Interpretation
The results strongly support our primary hypothesis. Condition A2 showed a statistically significant increase in outcome B compared to both the control condition (A1) and the alternative condition (A3).

## Limitations
- Sample was limited to a specific demographic
- Long-term effects not measured
- Potential confounding variable F not controlled

## Future Directions
Further research should explore the mechanism behind the observed effect and test generalizability across different populations.
"""
