"""Wrapper to run the judge evaluation from the correct directory."""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from llm_judge import run_judge_evaluation
run_judge_evaluation()
