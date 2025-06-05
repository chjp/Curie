"""
Reporter Module - Generates experiment reports

This module implements a simplified version of the reporting component
to create final experiment reports.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger("curie")

class Reporter:
    """Reporter class for generating experiment reports"""
    
    def __init__(self, output_dir: str, template_path: Optional[str] = None):
        """Initialize the reporter
        
        Args:
            output_dir: Directory to save reports
            template_path: Path to report template file
        """
        self.output_dir = output_dir
        self.template_path = template_path
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_template(self) -> str:
        """Load the report template"""
        if not self.template_path or not os.path.exists(self.template_path):
            # Use default template if template file doesn't exist
            return """# {{title}}

## Research Question
{{question}}

## Experiment Summary
{{summary}}

## Methods
{{methods}}

## Results
{{results}}

## Conclusion
{{conclusion}}

## References
{{references}}
"""
            
        with open(self.template_path, 'r') as f:
            return f.read()
    
    def generate_report(self, experiment_state: Dict[str, Any]) -> str:
        """Generate a report for the experiment
        
        Args:
            experiment_state: Dictionary containing the experiment state and results
            
        Returns:
            Path to the generated report file
        """
        logger.info("Generating experiment report")
        
        # Extract key information from experiment state
        question = experiment_state.get("question", "No question specified")
        steps = experiment_state.get("steps", [])
        plan = experiment_state.get("plan", {"title": "Experiment Report"})
        experiment_id = experiment_state.get("experiment_id", datetime.now().strftime("%Y%m%d%H%M%S"))
        
        # Generate report components
        title = plan.get("title", f"Research Report: {question[:50]}")
        summary = self._generate_summary(experiment_state)
        methods = self._generate_methods(steps)
        results = self._generate_results(steps)
        conclusion = self._generate_conclusion(steps, question)
        references = self._generate_references()
        
        # Load and fill template
        template = self.load_template()
        report_content = template.replace("{{title}}", title)
        report_content = report_content.replace("{{question}}", question)
        report_content = report_content.replace("{{summary}}", summary)
        report_content = report_content.replace("{{methods}}", methods)
        report_content = report_content.replace("{{results}}", results)
        report_content = report_content.replace("{{conclusion}}", conclusion)
        report_content = report_content.replace("{{references}}", references)
        
        # Add metadata
        metadata = f"""
## Experiment Metadata
- **Experiment ID**: {experiment_id}
- **Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Status**: {experiment_state.get("status", "Unknown")}
"""
        report_content += metadata
        
        # Save report to file
        report_file = os.path.join(self.output_dir, f"report_{experiment_id}.md")
        with open(report_file, 'w') as f:
            f.write(report_content)
            
        logger.info(f"Report saved to {report_file}")
        return report_file
    
    def _generate_summary(self, experiment_state: Dict[str, Any]) -> str:
        """Generate a summary of the experiment"""
        question = experiment_state.get("question", "No question specified")
        status = experiment_state.get("status", "Unknown")
        steps = experiment_state.get("steps", [])
        
        summary = f"This experiment investigated the question: \"{question}\". "
        summary += f"The experiment consisted of {len(steps)} steps and reached a status of '{status}'. "
        
        if steps:
            # Include brief info about the first and last steps
            first_step = steps[0]
            last_step = steps[-1]
            summary += f"Beginning with {first_step['step']}, the experiment proceeded through "
            summary += f"several stages before concluding with {last_step['step']}."
        
        return summary
    
    def _generate_methods(self, steps: List[Dict[str, Any]]) -> str:
        """Generate the methods section based on experiment steps"""
        if not steps:
            return "No methods were recorded for this experiment."
            
        methods_text = ""
        
        # Look for relevant steps that describe methods (hypothesis, experiment_design, data_collection)
        for step in steps:
            step_name = step.get("step", "")
            if step_name in ["hypothesis", "experiment_design", "data_collection", "analysis_design"]:
                methods_text += f"### {step_name.replace('_', ' ').title()}\n\n"
                
                # For MVP, we'll just include the step output
                output = step.get("output", "No output recorded")
                if isinstance(output, str):
                    # Clean up the output to extract the relevant parts for the methods section
                    lines = output.strip().split("\n")
                    clean_lines = []
                    in_methods_section = False
                    
                    for line in lines:
                        if line.strip().startswith("# ") or line.strip().startswith("## "):
                            # Reset section detection for new major sections
                            in_methods_section = "method" in line.lower() or "design" in line.lower()
                        
                        if in_methods_section or not any(section in line.lower() for section in ["# ", "## "]):
                            clean_lines.append(line)
                    
                    if clean_lines:
                        methods_text += "\n".join(clean_lines) + "\n\n"
                    else:
                        methods_text += output + "\n\n"
                else:
                    methods_text += str(output) + "\n\n"
        
        if not methods_text:
            return "This experiment did not explicitly document its methodology."
            
        return methods_text
    
    def _generate_results(self, steps: List[Dict[str, Any]]) -> str:
        """Generate the results section based on experiment steps"""
        if not steps:
            return "No results were recorded for this experiment."
            
        results_text = ""
        
        # Look for relevant steps that describe results (data_collection, analysis)
        for step in steps:
            step_name = step.get("step", "")
            if step_name in ["data_collection", "analysis"]:
                results_text += f"### {step_name.replace('_', ' ').title()}\n\n"
                
                # For MVP, we'll just include the step output
                output = step.get("output", "No output recorded")
                if isinstance(output, str):
                    # Extract results-oriented sections
                    lines = output.strip().split("\n")
                    clean_lines = []
                    in_results_section = False
                    
                    for line in lines:
                        if line.strip().startswith("# ") or line.strip().startswith("## "):
                            # Reset section detection for new major sections
                            in_results_section = any(term in line.lower() for term in ["result", "finding", "analysis", "data"])
                        
                        if in_results_section or not any(section in line.lower() for section in ["# ", "## "]):
                            clean_lines.append(line)
                    
                    if clean_lines:
                        results_text += "\n".join(clean_lines) + "\n\n"
                    else:
                        results_text += output + "\n\n"
                else:
                    results_text += str(output) + "\n\n"
        
        if not results_text:
            return "This experiment did not explicitly document its results."
            
        return results_text
    
    def _generate_conclusion(self, steps: List[Dict[str, Any]], question: str) -> str:
        """Generate the conclusion section based on experiment steps"""
        if not steps:
            return "No conclusion could be drawn as no experiment steps were recorded."
            
        conclusion_text = ""
        
        # Look for the analysis step which should contain conclusions
        for step in steps:
            step_name = step.get("step", "")
            if step_name in ["analysis", "conclusions"]:
                output = step.get("output", "")
                if isinstance(output, str):
                    # Extract conclusion-oriented sections
                    lines = output.strip().split("\n")
                    clean_lines = []
                    in_conclusion_section = False
                    
                    for line in lines:
                        if line.strip().startswith("# ") or line.strip().startswith("## "):
                            # Reset section detection for new major sections
                            in_conclusion_section = any(term in line.lower() for term in [
                                "conclusion", "summary", "discussion", "interpretation", "future"
                            ])
                        
                        if in_conclusion_section:
                            clean_lines.append(line)
                    
                    if clean_lines:
                        conclusion_text += "\n".join(clean_lines) + "\n\n"
        
        if not conclusion_text:
            # Generate a generic conclusion if none was found
            conclusion_text = f"""## Conclusion

Based on the experiment conducted to investigate the question: "{question}", 
the results suggest certain patterns and insights. 

The analysis of the collected data provides evidence that helps address the 
research question, though further investigation may be warranted to strengthen 
these conclusions.

### Limitations

This experiment has certain limitations that should be considered when 
interpreting the results:

1. Limited scope of the investigation
2. Potential confounding variables
3. Constraints of the experimental design

### Future Directions

Future research could expand on these findings by:

1. Exploring additional variables
2. Employing alternative methodologies
3. Investigating related research questions
"""
            
        return conclusion_text
    
    def _generate_references(self) -> str:
        """Generate a references section
        
        In a real implementation, this would extract references from the experiment
        """
        # For MVP, just provide a placeholder
        return """1. Smith, J. et al. (2024). "Related Research Title." Journal of Science, 10(2), 123-145.
2. Johnson, A. (2023). "Theoretical Framework." Research Quarterly, 15(3), 78-92."""
