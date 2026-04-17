"""
LLM as a Judge Evaluation for Speech-to-Speech Interactions

This module evaluates Speech-to-Speech (Nova Sonic) interactions based on telemetry data
retrieved from Amazon CloudWatch. It uses an LLM-as-Judge approach to assess conversation
quality and tool call accuracy.

## Main Classes

### S2SEvaluator
The primary class for running evaluations. Provides a unified interface for:
- Extracting traces from CloudWatch
- Processing spans into evaluation datasets
- Running LLM-as-a-Judge evaluations
- Generating evaluation reports
- Visualizing evaluation results

Usage in Jupyter notebooks:
```python
import llm_judge_text_eval_agent_core_observability as eval_module

# Initialize with existing boto3 session
evaluator = eval_module.S2SEvaluator(boto3_session=your_session)

# Extract traces
traces = evaluator.extract_traces_from_cloudwatch(hours_back=24)

# Process and evaluate
eval_data = evaluator.process_and_store_eval_dataset([traces], "output.jsonl")
config = evaluator.load_config("config.json")
validation_data = evaluator.load_validation_dataset("validation.jsonl")
judge = evaluator.initialize_judge()
results = evaluator.run_evaluation_iteration()
report = evaluator.generate_evaluation_report(results)
```

### LLMJudge
The evaluation engine using Amazon Bedrock models to judge conversation quality.

## Command-Line Usage
The script can also be run standalone:
```bash
python llm_judge_text_eval_agent_core_observability.py \\
    --hours-back 24 \\
    --num-runs 3 \\
    --category TechnicalInterview
```
"""

import argparse
import json
import os
import boto3
import time
import sys
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def set_debug_logging(enabled: bool = False):
    """Enable or disable debug logging."""
    level = logging.DEBUG if enabled else logging.INFO
    logger.setLevel(level)
    for handler in logging.root.handlers:
        handler.setLevel(level)

# Default paths
DEFAULT_CONFIG_PATH = "config/llm_judge_s2s_config.json"
DEFAULT_VALIDATION_DATASET_PATH = "data/s2s_validation_dataset.jsonl"
DEFAULT_EVAL_DATASET_PATH = "data/s2s_eval_data.jsonl"
DEFAULT_MAPPINGS_PATH = "config/manual_mappings.json"
DEFAULT_REPORT_OUTPUT_PATH = "./test-evaluation-results/"


def find_dotenv():
    """Find the .env file in various possible locations."""
    possible_paths = [
        Path('.env'),
        Path('..') / '.env',
        Path('..') / 'deployment' / '.env',
        Path(os.getcwd()) / '.env',
        Path(os.getcwd()).parent / '.env',
    ]

    for path in possible_paths:
        if path.exists():
            logger.info(f"Found .env file at: {path.absolute()}")
            return path

    logger.warning("No .env file found in expected locations")
    return None


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        raise


def load_validation_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load validation dataset from JSONL file."""
    validation_data = []
    try:
        with open(dataset_path, 'r') as f:
            for line in f:
                if line.strip():
                    validation_data.append(json.loads(line.strip()))
        logger.info(f"Loaded validation dataset from {dataset_path} with {len(validation_data)} entries")
        return validation_data
    except Exception as e:
        logger.error(f"Error loading validation dataset: {e}")
        return []


def load_eval_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load evaluation dataset from JSONL file."""
    eval_data = []
    try:
        with open(dataset_path, 'r') as f:
            for line in f:
                if line.strip():
                    eval_data.append(json.loads(line.strip()))
        logger.info(f"Loaded eval dataset from {dataset_path} with {len(eval_data)} entries")
        return eval_data
    except Exception as e:
        logger.error(f"Error loading eval dataset: {e}")
        return []


class S2SEvaluator:
    """
    Main class for Speech-to-Speech evaluation pipeline.
    Provides a unified interface for trace extraction, processing, and evaluation.
    """

    def __init__(self, boto3_session: Optional[boto3.Session] = None, region_name: str = 'us-east-1'):
        """
        Initialize the S2S Evaluator.

        Args:
            boto3_session: Optional existing boto3 session. If not provided, creates a new one.
            region_name: AWS region for services (default: us-east-1)
        """
        # Use provided session or create new one
        if boto3_session:
            self.session = boto3_session
            logger.info("Using provided boto3 session")
        else:
            env_path = find_dotenv()
            if env_path:
                load_dotenv(dotenv_path=env_path)
            self.session = boto3.Session(region_name=region_name)
            logger.info(f"Created new boto3 session for region: {region_name}")

        # Initialize clients
        self.cloudwatch_logs_client = self.session.client('logs')
        self.bedrock_runtime_client = self.session.client('bedrock-runtime')

        # State variables
        self.config = None
        self.validation_data = None
        self.eval_data = None
        self.judge = None

        logger.info("✅ S2SEvaluator initialized")

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        self.config = load_config(config_path)
        return self.config

    def load_validation_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load validation dataset from JSONL file."""
        self.validation_data = load_validation_dataset(dataset_path)
        return self.validation_data

    def load_eval_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load evaluation dataset from JSONL file."""
        self.eval_data = load_eval_dataset(dataset_path)
        return self.eval_data

    def load_manual_mappings(self, mappings_path: str) -> List[Dict[str, Any]]:
        """Load manual session-to-category mappings from JSON file."""
        manual_mappings = []
        try:
            with open(mappings_path, 'r') as f:
                manual_mappings = json.load(f)
            logger.info(f"Loaded {len(manual_mappings)} manual mappings from {mappings_path}")
            return manual_mappings
        except FileNotFoundError:
            logger.warning(f"No mappings file found at {mappings_path}")
            return []
        except Exception as e:
            logger.error(f"Error loading manual mappings from {mappings_path}: {e}")
            return []

    def extract_traces_from_cloudwatch(
        self,
        session_id: Optional[str] = None,
        hours_back: int = 24,
        log_group_name: str = 'aws/spans',
        limit: int = 10000
    ) -> Dict[str, Any]:
        """
        Query CloudWatch Logs for spans related to a session or all sessions.

        Args:
            session_id: Optional session ID. If None, retrieves all sessions.
            hours_back: Number of hours to look back (default: 24)
            log_group_name: CloudWatch log group name (default: 'aws/spans')
            limit: Maximum number of spans to retrieve (default: 10000)

        Returns:
            Dictionary containing session spans and metadata
        """
        return extract_traces_from_cloudwatch(
            session_id=session_id,
            hours_back=hours_back,
            log_group_name=log_group_name,
            region_name=self.session.region_name,
            limit=limit,
            logs_client=self.cloudwatch_logs_client
        )

    def process_and_store_eval_dataset(
        self,
        raw_traces: List[Dict[str, Any]],
        output_path: str
    ) -> List[Dict[str, Any]]:
        """
        Process raw CloudWatch spans into structured evaluation dataset.

        Args:
            raw_traces: List of raw trace data from CloudWatch
            output_path: Path to save the processed dataset

        Returns:
            List of processed session data
        """
        self.eval_data = process_and_store_eval_dataset(raw_traces, output_path)
        return self.eval_data

    def initialize_judge(self, config: Optional[Dict[str, Any]] = None) -> 'LLMJudge':
        """
        Initialize the LLM Judge with configuration.

        Args:
            config: Optional configuration dict. Uses self.config if not provided.

        Returns:
            Initialized LLMJudge instance
        """
        if config:
            self.config = config

        if not self.config:
            raise ValueError("Configuration not loaded. Call load_config() first or provide config.")

        self.judge = LLMJudge(self.config, boto3_session=self.session)
        return self.judge

    def run_evaluation_iteration(
        self,
        category_filter: Optional[str] = None,
        eval_data: Optional[List[Dict[str, Any]]] = None,
        validation_data: Optional[List[Dict[str, Any]]] = None,
        manual_mappings: Optional[List[Dict[str, Any]]] = None,
        auto_classifier=None,
    ) -> Dict[str, Any]:
        """
        Run a single evaluation iteration.

        Category resolution order:
          1. span metadata (set by test runners automatically)
          2. LLM auto-classifier (pass auto_classifier= to enable)
          3. manual mappings (legacy fallback)

        Args:
            category_filter: Optional category to filter by
            eval_data: Optional evaluation data (uses self.eval_data if not provided)
            validation_data: Optional validation data (uses self.validation_data if not provided)
            manual_mappings: Optional manual session-to-category mappings (legacy)
            auto_classifier: Optional SessionAutoClassifier instance

        Returns:
            Dictionary with evaluation results
        """
        if not self.judge:
            raise ValueError("Judge not initialized. Call initialize_judge() first.")

        eval_data = eval_data or self.eval_data
        validation_data = validation_data or self.validation_data

        if not eval_data or not validation_data:
            raise ValueError("Evaluation and validation data required. Load datasets first.")

        return run_evaluation_iteration(
            eval_data,
            validation_data,
            self.judge,
            category_filter,
            manual_mappings,
            auto_classifier=auto_classifier,
        )

    def build_auto_classifier(self, model_id: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"):
        """Create a SessionAutoClassifier pre-loaded with categories from the validation dataset.

        Call this after load_validation_dataset() and pass the result to
        run_evaluation_iteration(auto_classifier=...) to enable LLM-based
        category inference for sessions that have no span metadata.

        Args:
            model_id: Bedrock model to use for classification (default: Claude Haiku).

        Returns:
            SessionAutoClassifier instance
        """
        from auto_classifier import SessionAutoClassifier

        if not self.validation_data:
            raise ValueError("Validation data not loaded. Call load_validation_dataset() first.")

        known_categories = list({
            entry.get("category")
            for entry in self.validation_data
            if entry.get("category")
        })
        logger.info(f"Building auto-classifier with categories: {known_categories}")
        return SessionAutoClassifier(
            known_categories=known_categories,
            boto3_session=self.session,
            model_id=model_id,
        )

    def merge_results(self, run_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge results from multiple evaluation runs.

        Args:
            run_results: List of results from multiple runs

        Returns:
            Merged results dictionary
        """
        return merge_results(run_results)

    def generate_evaluation_report(self, merged_results: Dict[str, Any]) -> str:
        """
        Generate markdown evaluation report.

        Args:
            merged_results: Merged results from multiple runs

        Returns:
            Markdown formatted report string
        """
        return generate_evaluation_report(merged_results)

    def visualize_evaluation_results(
        self,
        merged_results: Optional[Dict[str, Any]] = None,
        evaluation_output_dir: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Generate visualizations for evaluation results.

        Args:
            merged_results: Optional merged results dict. If not provided, reads from evaluation_output_dir.
            evaluation_output_dir: Optional path to evaluation results directory. If not provided, uses latest.
            save_path: Optional path to save visualizations. If not provided, saves to evaluation_output_dir.

        Returns:
            None (displays and saves visualizations)
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd

        # Load results if not provided
        if merged_results is None:
            if evaluation_output_dir is None:
                # Find the latest evaluation directory
                results_base_dir = Path(DEFAULT_REPORT_OUTPUT_PATH)
                if not results_base_dir.exists():
                    raise ValueError(f"Results directory not found: {results_base_dir}")

                eval_dirs = sorted(results_base_dir.glob("evaluation_report_*"))
                if not eval_dirs:
                    raise ValueError(f"No evaluation reports found in {results_base_dir}")

                evaluation_output_dir = str(eval_dirs[-1])
                logger.info(f"Using latest evaluation directory: {evaluation_output_dir}")

            # Load complete_results.json
            results_file = Path(evaluation_output_dir) / "complete_results.json"
            if not results_file.exists():
                raise ValueError(f"Results file not found: {results_file}")

            with open(results_file, 'r') as f:
                merged_results = json.load(f)
            logger.info(f"Loaded results from {results_file}")

        # Set default save path
        if save_path is None and evaluation_output_dir:
            save_path = str(Path(evaluation_output_dir) / 'evaluation_visualizations.png')
        elif save_path is None:
            save_path = 'evaluation_visualizations.png'

        # Extract data for visualization
        all_results = merged_results.get('all_results', [])
        eval_datasets = merged_results.get('eval_datasets', [[]])[0] if merged_results.get('eval_datasets') else []

        # Set style
        sns.set_theme(style="whitegrid")

        # Create figure with subplots (3 rows, 3 columns for additional visualizations)
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle(f'Evaluation Results Summary', fontsize=16, fontweight='bold')

        # 1. Successful Evaluations by Category
        if all_results:
            # Count successful evaluations (no error) by category
            success_by_category = {}
            for result in all_results:
                if 'error' not in result:
                    cat = result.get('category', 'Unknown')
                    success_by_category[cat] = success_by_category.get(cat, 0) + 1

            if success_by_category:
                categories = list(success_by_category.keys())
                counts = list(success_by_category.values())

                axes[0, 0].barh(categories, counts, color='#2ecc71')
                axes[0, 0].set_xlabel('Number of Successful Evaluations')
                axes[0, 0].set_title('Successful Evaluations by Category')
                axes[0, 0].tick_params(axis='y', labelsize=9)

                # Add value labels on bars
                for i, v in enumerate(counts):
                    axes[0, 0].text(v + 0.1, i, str(v), va='center', fontsize=9)
            else:
                axes[0, 0].text(0.5, 0.5, 'No successful evaluations', ha='center', va='center')
                axes[0, 0].set_title('Successful Evaluations by Category')
        else:
            axes[0, 0].text(0.5, 0.5, 'No data', ha='center', va='center')
            axes[0, 0].set_title('Successful Evaluations by Category')

        # 2. Evaluation Status Overview (Success vs Error)
        error_count = sum(1 for r in all_results if 'error' in r)
        success_count = len(all_results) - error_count

        if len(all_results) > 0:
            axes[0, 1].pie([success_count, error_count], labels=['Success', 'Error'],
                          autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], startangle=90)
            axes[0, 1].set_title(f'Evaluation Status Overview\n(Total: {len(all_results)} evaluations)')
        else:
            axes[0, 1].text(0.5, 0.5, 'No data', ha='center', va='center')
            axes[0, 1].set_title('Evaluation Status Overview')

        # 3. Average scores by category
        if all_results:
            category_scores = {}
            for result in all_results:
                if 'error' not in result and 'overall' in result and 'score' in result['overall']:
                    cat = result.get('category', 'Unknown')
                    if cat not in category_scores:
                        category_scores[cat] = []
                    category_scores[cat].append(result['overall']['score'])

            if category_scores:
                categories = list(category_scores.keys())
                avg_scores = [sum(scores) / len(scores) for scores in category_scores.values()]

                # Color bars based on score (green for good, yellow for medium, red for low)
                colors = ['#e74c3c' if score < 0.5 else '#f39c12' if score < 0.7 else '#2ecc71' for score in avg_scores]

                axes[0, 2].barh(categories, avg_scores, color=colors)
                axes[0, 2].set_xlabel('Average Score (0-1 scale)')
                axes[0, 2].set_title('Average Overall Score by Category')
                axes[0, 2].set_xlim(0, 1.0)  # Score is 0 or 1
                axes[0, 2].tick_params(axis='y', labelsize=9)

                # Add value labels on bars - position them to be visible even for zero values
                for i, v in enumerate(avg_scores):
                    if v > 0:
                        axes[0, 2].text(v + 0.02, i, f'{v:.2f}', va='center', fontsize=9)
                    else:
                        # For zero values, place text to the right with a visible position
                        axes[0, 2].text(0.05, i, f'{v:.2f}', va='center', fontsize=9, color='#e74c3c')
            else:
                axes[0, 2].text(0.5, 0.5, 'No score data', ha='center', va='center')
                axes[0, 2].set_title('Average Overall Score by Category')
        else:
            axes[0, 2].text(0.5, 0.5, 'No data', ha='center', va='center')
            axes[0, 2].set_title('Average Overall Score by Category')

        # 4. Token usage by session
        if eval_datasets and len(eval_datasets) > 0:
            sessions_df = pd.DataFrame([{
                'sessionId': s['sessionId'][:8] + '...',
                'total_tokens': s.get('total_tokens') or 0,
                'input_tokens': s.get('input_tokens') or 0,
                'output_tokens': s.get('output_tokens') or 0
            } for s in eval_datasets[:10]])  # Limit to first 10 for readability

            if not sessions_df.empty:
                x_pos = range(len(sessions_df))
                axes[1, 0].bar(x_pos, sessions_df['input_tokens'], label='Input', color='#3498db')
                axes[1, 0].bar(x_pos, sessions_df['output_tokens'], bottom=sessions_df['input_tokens'],
                             label='Output', color='#9b59b6')
                axes[1, 0].set_title(f'Token Usage by Session\n(Showing first {len(sessions_df)} sessions)')
                axes[1, 0].set_xlabel('Session ID')
                axes[1, 0].set_ylabel('Tokens')
                axes[1, 0].set_xticks(x_pos)
                axes[1, 0].set_xticklabels(sessions_df['sessionId'], rotation=45, ha='right', fontsize=8)
                axes[1, 0].legend()
        else:
            axes[1, 0].text(0.5, 0.5, 'No session data', ha='center', va='center')
            axes[1, 0].set_title('Token Usage by Session')

        # 5. Cost by session
        if eval_datasets and len(eval_datasets) > 0:
            costs = [s.get('cost') or 0 for s in eval_datasets[:10]]
            session_ids = [s['sessionId'][:8] + '...' for s in eval_datasets[:10]]

            if costs:
                axes[1, 1].bar(range(len(costs)), costs, color='#f39c12')
                axes[1, 1].set_title(f'Cost by Session\n(Showing first {len(costs)} sessions)')
                axes[1, 1].set_xlabel('Session ID')
                axes[1, 1].set_ylabel('Cost (USD)')
                axes[1, 1].set_xticks(range(len(costs)))
                axes[1, 1].set_xticklabels(session_ids, rotation=45, ha='right', fontsize=8)
                axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'${y:.4f}'))
        else:
            axes[1, 1].text(0.5, 0.5, 'No cost data', ha='center', va='center')
            axes[1, 1].set_title('Cost by Session')

        # 6. Duration by session
        if eval_datasets and len(eval_datasets) > 0:
            durations = [s.get('duration') or 0 for s in eval_datasets[:10]]
            session_ids = [s['sessionId'][:8] + '...' for s in eval_datasets[:10]]

            if durations:
                axes[1, 2].bar(range(len(durations)), durations, color='#1abc9c')
                axes[1, 2].set_title(f'Session Duration\n(Showing first {len(durations)} sessions)')
                axes[1, 2].set_xlabel('Session ID')
                axes[1, 2].set_ylabel('Duration (seconds)')
                axes[1, 2].set_xticks(range(len(durations)))
                axes[1, 2].set_xticklabels(session_ids, rotation=45, ha='right', fontsize=8)
        else:
            axes[1, 2].text(0.5, 0.5, 'No duration data', ha='center', va='center')
            axes[1, 2].set_title('Session Duration')

        # 7. Average duration by category (NEW)
        if all_results and eval_datasets:
            # Build mapping of session_id to category from results
            session_to_category = {r.get('session_id'): r.get('category') for r in all_results if 'error' not in r}

            # Calculate average duration by category
            category_durations = {}
            for session in eval_datasets:
                session_id = session.get('sessionId')
                category = session_to_category.get(session_id)
                duration = session.get('duration') or 0
                if category and duration:
                    if category not in category_durations:
                        category_durations[category] = []
                    category_durations[category].append(duration)

            if category_durations:
                categories = list(category_durations.keys())
                avg_durations = [sum(durations) / len(durations) for durations in category_durations.values()]

                axes[2, 0].barh(categories, avg_durations, color='#e74c3c')
                axes[2, 0].set_xlabel('Average Duration (seconds)')
                axes[2, 0].set_title('Average Session Duration by Category')
                axes[2, 0].tick_params(axis='y', labelsize=9)

                # Add value labels on bars
                for i, v in enumerate(avg_durations):
                    axes[2, 0].text(v + 0.5, i, f'{v:.1f}s', va='center', fontsize=9)
            else:
                axes[2, 0].text(0.5, 0.5, 'No duration data', ha='center', va='center')
                axes[2, 0].set_title('Average Session Duration by Category')
        else:
            axes[2, 0].text(0.5, 0.5, 'No data', ha='center', va='center')
            axes[2, 0].set_title('Average Session Duration by Category')

        # 8. Summary statistics
        summary_text = f"""Total Evaluations: {merged_results.get('total_evaluations', 0)}
Total Runs: {merged_results.get('runs', 0)}
Success Rate: {success_count}/{len(all_results)} ({success_count/len(all_results)*100:.1f}%)
Total Tokens of Evaluated Sessions: {merged_results.get('total_tokens', 0):,}
Total Cost of Evaluated Sessions: ${merged_results.get('total_cost', 0):.4f}

Sessions in Dataset: {len(eval_datasets)}
Evaluated Sessions: {success_count}
Failed Evaluations: {error_count}"""

        axes[2, 1].text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                       fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        axes[2, 1].axis('off')
        axes[2, 1].set_title('Summary Statistics')

        # 9. Hide the unused subplot
        axes[2, 2].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Visualizations saved to {save_path}")
        plt.show()

        return save_path


def extract_traces_from_cloudwatch(
    session_id: Optional[str] = None,
    hours_back: int = 24,
    log_group_name: str = 'aws/spans',
    region_name: str = 'us-east-1',
    limit: int = 10000,
    logs_client: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Query CloudWatch Logs for spans related to a session or all sessions.
    If session_id is None, retrieves all sessions from the specified time period.
    Extract root span and child spans.
    """
    try:
        end_time = datetime.now(timezone.utc).replace(tzinfo=None)
        start_time = end_time - timedelta(hours=hours_back)

        if session_id:
            logger.info(f"Searching CloudWatch for session {session_id}")
        else:
            logger.info(f"Searching CloudWatch for all sessions from the last {hours_back} hour(s)")
        logger.info(f"Time range: {start_time} to {end_time}")
        logger.info(f"CloudWatch region: {region_name}")
        logger.info(f"CloudWatch log group: {log_group_name}")

        # Use provided client or create new one
        if logs_client is None:
            logs_client = boto3.client('logs', region_name=region_name)

        # List available log groups for debugging
        try:
            log_groups_response = logs_client.describe_log_groups()
            available_groups = [lg['logGroupName'] for lg in log_groups_response.get('logGroups', [])]
            if available_groups:
                logger.debug(f"Available log groups: {available_groups}")
            else:
                logger.warning("No log groups found in CloudWatch!")
        except Exception as e:
            logger.debug(f"Could not list log groups: {e}")

        # Use filter_log_events API for better compatibility with structured span logs
        logger.info("Using filter_log_events API to retrieve spans...")

        processed_spans = []
        next_token = None
        events_scanned = 0

        while True:
            try:
                # Build filter pattern
                filter_pattern = None
                if session_id:
                    # Use JSON filter pattern for CloudWatch Logs to find specific session
                    filter_pattern = f'{{ $.attributes.session_id = "{session_id}" }}'
                    logger.debug(f"Using filter pattern for session: {filter_pattern}")
                else:
                    # When getting all sessions, don't use filter pattern
                    # Filter in Python after retrieval for better compatibility
                    logger.debug("No filter pattern - will retrieve all events and filter in Python")

                params = {
                    'logGroupName': log_group_name,
                    'startTime': int(start_time.timestamp() * 1000),  # milliseconds
                    'endTime': int(end_time.timestamp() * 1000),
                    'interleaved': True
                }

                if filter_pattern:
                    params['filterPattern'] = filter_pattern
                if next_token:
                    params['nextToken'] = next_token

                logger.debug(f"Calling filter_log_events (limit {limit}, already scanned {events_scanned})...")
                response = logs_client.filter_log_events(**params)

                events = response.get('events', [])
                logger.info(f"Retrieved {len(events)} events in this batch")

                # Track total events scanned across all batches
                events_scanned += len(events)

                if not events:
                    logger.info("No more events in this batch")
                    break

                # Parse JSON messages from events - continue until we have enough PARSED spans
                for idx, event in enumerate(events):
                    # Stop if we've reached the limit of PARSED spans (not raw events)
                    if len(processed_spans) >= limit:
                        logger.info(f"Reached limit of {limit} parsed spans")
                        break

                    try:
                        message = event.get('message', '')
                        if message:
                            span_data = json.loads(message)

                            # Filter by session_id if specified
                            if session_id:
                                span_session_id = span_data.get('attributes', {}).get('session_id') or \
                                                 span_data.get('attributes', {}).get('session.id')
                                if span_session_id != session_id:
                                    logger.debug(f"Skipping span with session_id={span_session_id}")
                                    continue

                            processed_spans.append(span_data)
                            logger.debug(f"Parsed span {len(processed_spans)}: {span_data.get('name', 'unknown')}")
                    except json.JSONDecodeError as e:
                        logger.debug(f"Failed to parse event {idx} as JSON: {e}")
                        continue

                # Check if we've reached the span limit
                if len(processed_spans) >= limit:
                    logger.info(f"Reached limit of {limit} parsed spans")
                    break

                # Check if there are more events to retrieve
                next_token = response.get('nextToken')
                if not next_token:
                    logger.info(f"No more events to retrieve (scanned {events_scanned} events total)")
                    break

            except Exception as e:
                logger.error(f"Error retrieving spans: {e}")
                if len(processed_spans) > 0:
                    logger.info(f"Continuing with {len(processed_spans)} spans already retrieved")
                    break
                raise

        logger.info(f"Total spans retrieved: {len(processed_spans)}")

        # Group spans by session if retrieving all sessions
        if session_id:
            logger.info(f"Retrieved {len(processed_spans)} spans for session {session_id}")
            return {
                'session_id': session_id,
                'spans': processed_spans,
                'query_status': 'Complete',
                'total_spans': len(processed_spans)
            }
        else:
            # Group by session ID
            sessions_dict = {}
            spans_without_session_id = 0

            for span_idx, span in enumerate(processed_spans):
                # Try multiple possible paths for session ID
                sid = span.get('attributes', {}).get('session_id')
                if not sid:
                    sid = span.get('attributes', {}).get('session.id')
                if not sid:
                    sid = span.get('attributes', {}).get('aws.session.id')

                if sid:
                    if sid not in sessions_dict:
                        sessions_dict[sid] = []
                    sessions_dict[sid].append(span)
                    logger.debug(f"Span {span_idx} grouped into session: {sid}")
                else:
                    spans_without_session_id += 1
                    # Log first span structure if no session ID found
                    if spans_without_session_id == 1:
                        logger.warning(f"Span {span_idx} has no session ID. Available attributes keys: {list(span.get('attributes', {}).keys())}")
                        logger.debug(f"Sample span structure: {json.dumps(span, indent=2, default=str)[:500]}")

            logger.info(f"Session grouping: {len(sessions_dict)} sessions, {spans_without_session_id} spans without session ID")

            logger.info(f"Retrieved {len(processed_spans)} spans across {len(sessions_dict)} sessions")
            return {
                'session_id': 'all',
                'sessions': sessions_dict,
                'query_status': 'Complete',
                'total_spans': len(processed_spans),
                'total_sessions': len(sessions_dict)
            }

    except Exception as e:
        logger.error(f"Error querying CloudWatch: {str(e)}")
        raise


def process_and_store_eval_dataset(
    raw_traces: List[Dict[str, Any]],
    output_path: str
) -> List[Dict[str, Any]]:
    """
    Process raw CloudWatch spans into structured evaluation dataset.
    Extract root span and parse child spans.
    Handles both individual session traces and grouped multi-session traces.
    """
    processed_sessions = []

    for trace in raw_traces:
        # Handle both single session and multi-session traces
        if trace.get('session_id') == 'all' and 'sessions' in trace:
            # Multi-session trace from retrieving all sessions
            for session_id, session_spans in trace.get('sessions', {}).items():
                session_data = _process_session_spans(session_id, session_spans)
                if session_data:
                    processed_sessions.append(session_data)
        else:
            # Single session trace
            session_id = trace.get('session_id')
            spans = trace.get('spans', [])
            session_data = _process_session_spans(session_id, spans)
            if session_data:
                processed_sessions.append(session_data)

    # Save to JSONL
    try:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w') as f:
            for session in processed_sessions:
                f.write(json.dumps(session) + '\n')
        logger.info(f"Saved {len(processed_sessions)} processed sessions to {output_path}")
    except Exception as e:
        logger.error(f"Error saving eval dataset: {e}")

    return processed_sessions


def _process_session_spans(
    session_id: str,
    spans: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Helper function to process spans for a single session.
    Extracts only required fields and builds conversation turns.
    Handles both old text-based spans (userInput, assistantOutput) and new audio-based spans (audioInput, audioOutput).
    """
    root_span = None
    child_spans = []

    # Separate root span from child spans
    for span in spans:
        if span.get('attributes', {}).get('aws.span.kind') == 'LOCAL_ROOT':
            root_span = span
        else:
            child_spans.append(span)

    # Extract root span metadata
    model_id = None
    total_tokens = 0
    output_tokens = 0
    input_tokens = 0
    cost = 0.0
    currency = ''
    start_time = None
    end_time = None
    duration = None

    if root_span:
        attrs = root_span.get('attributes', {})
        model_id = attrs.get('model.id')
        total_tokens = attrs.get('total_tokens', 0)
        output_tokens = attrs.get('output_tokens', 0)
        input_tokens = attrs.get('input_tokens', 0)
        cost = attrs.get('cost', 0.0)
        currency = attrs.get('currency', 'USD')

        # Extract timing information - handle both nanoseconds and milliseconds
        start_time_nano = root_span.get('startTimeUnixNano', 0)
        end_time_nano = root_span.get('endTimeUnixNano', 0)
        duration_nano = root_span.get('durationNano', 0)
        if start_time_nano:
            # If value is already in milliseconds range (< 1e12), use as-is; otherwise convert from nanoseconds
            if start_time_nano > 1e12:
                start_time_ms = start_time_nano / 1e6  # Convert nanoseconds to milliseconds
            else:
                start_time_ms = start_time_nano
            # Convert Unix timestamp (ms) to ISO 8601 datetime string
            start_time = datetime.fromtimestamp(start_time_ms / 1000, tz=timezone.utc).isoformat(timespec='milliseconds')
        if end_time_nano:
            if end_time_nano > 1e12:
                end_time_ms = end_time_nano / 1e6  # Convert nanoseconds to milliseconds
            else:
                end_time_ms = end_time_nano
            # Convert Unix timestamp (ms) to ISO 8601 datetime string
            end_time = datetime.fromtimestamp(end_time_ms / 1000, tz=timezone.utc).isoformat(timespec='milliseconds')
        if duration_nano:
            duration = duration_nano / 1e9  # Convert nanoseconds to seconds

    # Parse child spans into conversation turns
    turns = []
    current_turn = {}
    last_message_type = None  # Track the last message type to detect turn boundaries

    for span in child_spans:
        span_name = span.get('name')
        attrs = span.get('attributes', {})

        # Skip audio spans and session markers, only process text-based conversation spans
        if span_name == 'audioInput' or span_name == 'audioOutput' or span_name == 'contentEnd' or span_name == 'sessionStart' or span_name == 'sessionEnd':
            logger.debug(f"Skipping non-conversation span: {span_name}")
            continue

        if span_name == 'systemPrompt':
            # Try multiple possible attribute locations for system prompt content
            content = attrs.get('input', '') or attrs.get('output', '') or attrs.get('output.content', '')
            if content:
                current_turn['systemPrompt'] = content
            logger.debug(f"Parsed systemPrompt: {content[:50] if content else 'empty'}")

        elif span_name == 'userInput':
            # If we were processing a different type, finalize the turn and start a new one
            if last_message_type and last_message_type != 'user':
                if current_turn:
                    turns.append(current_turn)
                current_turn = {}

            if 'user' not in current_turn:
                current_turn['user'] = ''

            # Try multiple possible attribute locations for user input content
            content = attrs.get('output', '') or attrs.get('output.content', '') or attrs.get('input', '')
            if current_turn['user'] and content:
                current_turn['user'] += ' ' + content
            else:
                current_turn['user'] += content
            last_message_type = 'user'
            logger.debug(f"Parsed userInput: {content[:50] if content else 'empty'}")

        elif span_name == 'assistantOutput':
            # If we were processing a different type, finalize the turn and start a new one
            if last_message_type and last_message_type != 'assistant':
                if current_turn:
                    turns.append(current_turn)
                current_turn = {}

            if 'assistant' not in current_turn:
                current_turn['assistant'] = ''

            # Try multiple possible attribute locations for assistant output content
            content = attrs.get('output', '') or attrs.get('output.content', '') or attrs.get('input', '')
            if current_turn['assistant'] and content:
                current_turn['assistant'] += ' ' + content
            else:
                current_turn['assistant'] += content
            last_message_type = 'assistant'
            logger.debug(f"Parsed assistantOutput: {content[:50] if content else 'empty'}")

        elif span_name == 'toolUse' or span_name == 'agentToolUse':
            # Tool calls should be added to current turn, but mark as a state change
            if 'tools' not in current_turn:
                current_turn['tools'] = []
            tool_info = {
                'tool_name': attrs.get('input.toolName'),
                'tool_run_time': attrs.get('tool_run_time'),
                'params': attrs.get('input.params'),
                'result': attrs.get('output.result')
            }
            current_turn['tools'].append(tool_info)
            last_message_type = 'tool'  # Mark that a tool was called

    if current_turn:
        turns.append(current_turn)

    # Extract category injected by test runners (Option A — metadata injection)
    span_category = None
    if root_span:
        span_category = root_span.get('attributes', {}).get('session.category')
    if not span_category:
        # Also check child spans (sessionStart span carries the attribute too)
        for span in child_spans:
            span_category = span.get('attributes', {}).get('session.category')
            if span_category:
                break

    # Build clean session data with only required fields
    session_data = {
        'sessionId': session_id,
        'model_id': model_id,
        'total_tokens': total_tokens,
        'output_tokens': output_tokens,
        'input_tokens': input_tokens,
        'cost': cost,
        'currency': currency,
        'startTime': start_time,
        'endTime': end_time,
        'duration': duration,
        'turns': turns,
    }
    if span_category:
        session_data['category'] = span_category

    return session_data


class LLMJudge:
    """LLM-as-Judge for evaluating speech-to-speech interactions."""

    def __init__(self, config: Dict[str, Any], boto3_session: Optional[boto3.Session] = None):
        self.config = config
        self.judge_model_id = config.get('judge_model', {}).get('id')
        self.max_tokens = config.get('judge_model', {}).get('max_tokens', 1024)
        self.temperature = config.get('judge_model', {}).get('temperature', 0.0)
        self.evaluation_criteria = config.get('evaluation_criteria', {})
        self.prompt_template = config.get('prompt_template', '')

        # Use provided session or create new one
        if boto3_session:
            self.bedrock_client = boto3_session.client('bedrock-runtime')
        else:
            env_path = find_dotenv()
            if env_path:
                load_dotenv(dotenv_path=env_path)
            self.bedrock_client = boto3.client(
                service_name='bedrock-runtime',
                region_name=os.getenv('AWS_REGION', 'us-east-1')
            )

        logger.info(f"Initialized LLMJudge with model: {self.judge_model_id}")

    def evaluate_session(self,
                        session_data: Dict[str, Any],
                        validation_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a session against validation criteria."""
        try:
            prompt = self._create_evaluation_prompt(session_data, validation_entry)
            evaluation_result = self._call_judge_model(prompt)
            logger.debug(f"Raw evaluation result: {evaluation_result[:200]}")
            parsed_result = self._parse_evaluation_result(evaluation_result)

            parsed_result['session_id'] = session_data.get('sessionId')
            parsed_result['category'] = validation_entry.get('category')
            parsed_result['model_id'] = session_data.get('model_id')
            parsed_result['total_tokens'] = session_data.get('total_tokens')
            parsed_result['cost'] = session_data.get('cost')

            return parsed_result

        except Exception as e:
            logger.error(f"Error evaluating session: {e}")
            logger.debug(f"Full error trace:", exc_info=True)
            return {
                'error': str(e),
                'session_id': session_data.get('sessionId'),
                'category': validation_entry.get('category')
            }

    def _create_evaluation_prompt(self,
                                 session_data: Dict[str, Any],
                                 validation_entry: Dict[str, Any]) -> str:
        """Create evaluation prompt."""
        criteria_text = "\n".join([f"- {name}: {desc}"
                                   for name, desc in self.evaluation_criteria.items()])

        turns = session_data.get('turns', [])
        turns_text = json.dumps(turns, indent=2) if turns else "None"

        prompt = self.prompt_template.format(
            expected_conversation_flow=validation_entry.get('turns', ''),
            actual_conversation_flow=turns_text,
            criteria_text=criteria_text
        )

        return prompt

    def _call_judge_model(self, prompt: str) -> str:
        """Call Bedrock with evaluation prompt."""
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature
        }

        response = self.bedrock_client.invoke_model(
            modelId=self.judge_model_id,
            body=json.dumps(request_body)
        )

        response_body = json.loads(response.get('body').read())
        return response_body.get('content', [{}])[0].get('text', '')

    def _parse_evaluation_result(self, result_text: str) -> Dict[str, Any]:
        """Parse JSON evaluation result."""
        try:
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = result_text[json_start:json_end]
                return json.loads(json_str)
            else:
                return json.loads(result_text)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse evaluation result: {e}")
            return {"error": "Failed to parse evaluation result", "raw": result_text}


def run_evaluation_iteration(
    eval_data: List[Dict[str, Any]],
    validation_data: List[Dict[str, Any]],
    judge: LLMJudge,
    category_filter: Optional[str] = None,
    manual_mappings: Optional[List[Dict[str, Any]]] = None,
    auto_classifier=None,
) -> Dict[str, Any]:
    """
    Run single evaluation iteration.

    Category resolution order (hybrid approach):
      1. span metadata  — 'category' field set by test runners via session.category span attribute
      2. LLM classifier — auto_classifier.classify() for sessions without span metadata
      3. manual mappings — legacy {sessionId, category} pairs from config/manual_mappings.json

    Args:
        eval_data: Sessions extracted from CloudWatch (have sessionId, may have category)
        validation_data: Category templates (have category, no sessionId)
        judge: LLM judge instance
        category_filter: Optional category to filter by
        manual_mappings: Optional list of {sessionId, category} mappings (legacy fallback)
        auto_classifier: Optional SessionAutoClassifier instance (LLM fallback)

    Returns:
        Dictionary with evaluation results
    """
    results = []
    total_tokens = 0
    total_cost = 0.0

    # Build manual mappings lookup (legacy fallback)
    mappings_dict = {}
    if manual_mappings:
        for mapping in manual_mappings:
            sid = mapping.get('sessionId')
            cat = mapping.get('category')
            if sid and cat:
                mappings_dict[sid] = cat
        logger.info(f"Loaded {len(mappings_dict)} manual mappings (legacy fallback)")

    # Build validation data lookup by category
    validation_by_category = {}
    for val_entry in validation_data:
        category = val_entry.get('category')
        if category:
            validation_by_category[category] = val_entry

    logger.info(f"Available validation categories: {list(validation_by_category.keys())}")

    sessions_evaluated = 0
    sessions_skipped = 0

    for session in eval_data:
        session_id = session.get('sessionId')

        # ── Tier 1: span metadata (injected by test runners) ──────────────
        category = session.get('category')
        if category:
            logger.debug(f"Session {session_id}: category from span metadata → {category}")

        # ── Tier 2: LLM auto-classifier ───────────────────────────────────
        if not category and auto_classifier:
            category = auto_classifier.classify(session)
            if category:
                logger.info(f"Session {session_id}: category from LLM classifier → {category}")

        # ── Tier 3: manual mappings (legacy) ──────────────────────────────
        if not category:
            category = mappings_dict.get(session_id)
            if category:
                logger.debug(f"Session {session_id}: category from manual mappings → {category}")

        if not category:
            logger.debug(f"No category found for session {session_id}, skipping")
            sessions_skipped += 1
            continue

        # Find the validation entry for this category
        validation_entry = validation_by_category.get(category)
        if not validation_entry:
            logger.warning(f"Category '{category}' has no validation template — skipping session {session_id}")
            sessions_skipped += 1
            continue

        # Apply optional category filter
        if category_filter and category != category_filter:
            sessions_skipped += 1
            continue

        logger.info(f"Evaluating session {session_id} → category '{category}'")

        evaluation = judge.evaluate_session(session, validation_entry)
        results.append(evaluation)
        sessions_evaluated += 1

        total_tokens += session.get('total_tokens', 0)
        total_cost += float(session.get('cost', 0)) if session.get('cost') else 0.0

        time.sleep(1)  # Rate limiting

    logger.info(f"Evaluation complete: {sessions_evaluated} evaluated, {sessions_skipped} skipped")

    if sessions_evaluated == 0 and not mappings_dict and not auto_classifier:
        logger.warning(
            "No sessions were evaluated. Sessions from new test runs include span metadata "
            "automatically. For older sessions, provide manual_mappings or an auto_classifier."
        )

    return {
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'total_tokens': total_tokens,
        'total_cost': total_cost,
        'evaluations_count': len(results),
        'eval_dataset': eval_data
    }


def merge_results(run_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge results from multiple runs."""
    all_results = []
    total_tokens = 0
    total_cost = 0.0
    eval_datasets = []

    for run in run_results:
        all_results.extend(run.get('results', []))
        total_tokens += run.get('total_tokens', 0)
        total_cost += run.get('total_cost', 0.0)
        if 'eval_dataset' in run:
            eval_datasets.append(run.get('eval_dataset', []))

    return {
        'runs': len(run_results),
        'total_evaluations': len(all_results),
        'all_results': all_results,
        'total_tokens': total_tokens,
        'total_cost': total_cost,
        'eval_datasets': eval_datasets
    }


def generate_evaluation_report(merged_results: Dict[str, Any]) -> str:
    """Generate markdown evaluation report."""
    results = merged_results.get('all_results', [])

    if not results:
        return "# Evaluation Report\n\nNo results to report."

    # Calculate average scores
    criteria_scores = {}
    overall_scores = []

    for result in results:
        if 'criteria' in result:
            for criterion, data in result.get('criteria', {}).items():
                if isinstance(data, dict) and 'score' in data:
                    if criterion not in criteria_scores:
                        criteria_scores[criterion] = []
                    criteria_scores[criterion].append(data['score'])

        if 'overall' in result and 'score' in result['overall']:
            overall_scores.append(result['overall']['score'])

    # Generate report
    report = []
    report.append("# Speech-to-Speech Evaluation Report")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total evaluations: {len(results)}")
    report.append(f"Evaluation runs: {merged_results.get('runs', 1)}")
    report.append("")

    # Input Dataset Information
    eval_datasets = merged_results.get('eval_datasets', [])
    if eval_datasets:
        report.append("## Input Dataset Information")
        for run_idx, eval_dataset in enumerate(eval_datasets, 1):
            report.append(f"### Run {run_idx}")

            # Breakdown by category - get from results which have category from validation data
            category_counts = {}
            for result in results:
                cat = result.get('category', 'Unknown')
                category_counts[cat] = category_counts.get(cat, 0) + 1

            # Show count of evaluated sessions (sessions with mappings), not all sessions
            num_evaluated = sum(category_counts.values())
            report.append(f"- Total evaluated samples (with mappings): {num_evaluated}")
            report.append(f"- Total sessions in CloudWatch: {len(eval_dataset)}")
            report.append(f"- Samples without mappings: {len(eval_dataset) - num_evaluated}")

            if category_counts:
                report.append("- Samples by category:")
                for cat in sorted(category_counts.keys()):
                    report.append(f"  - {cat}: {category_counts[cat]}")

            # Aggregate metrics from ONLY the evaluated sessions (not all sessions)
            evaluated_session_ids = {result.get('session_id') for result in results}
            evaluated_sessions = [s for s in eval_dataset if s.get('sessionId') in evaluated_session_ids]

            total_tokens = sum(session.get('total_tokens', 0) for session in evaluated_sessions)
            total_cost = sum(float(session.get('cost', 0)) if session.get('cost') else 0.0 for session in evaluated_sessions)
            report.append(f"- Total tokens (evaluated sessions only): {total_tokens}")
            report.append(f"- Total cost (evaluated sessions only): ${total_cost:.4f}")
            report.append("")

        report.append("")

    # Summary statistics
    report.append("## Input Evaluation Dataset Summary")
    report.append(f"Total tokens in evaluated sessions: {merged_results.get('total_tokens', 0)}")
    report.append(f"Total cost of evaluated sessions: ${merged_results.get('total_cost', 0.0):.4f}")
    report.append("*Note: These metrics reflect the input evaluation dataset, not the cost of running this evaluation.*")
    report.append("")

    # Average scores
    report.append("## Average Scores by Criterion")
    report.append("| Criterion | Average Score |")
    report.append("|-----------|---------------|")

    for criterion, scores in sorted(criteria_scores.items()):
        if scores:
            avg = sum(scores) / len(scores)
            report.append(f"| {criterion} | {avg:.2f} |")

    if overall_scores:
        avg_overall = sum(overall_scores) / len(overall_scores)
        report.append(f"| **Overall** | **{avg_overall:.2f}** |")

    report.append("")

    # Category breakdown
    results_by_category = {}
    for result in results:
        cat = result.get('category', 'Unknown')
        if cat not in results_by_category:
            results_by_category[cat] = []
        results_by_category[cat].append(result)

    report.append("## Results by Category")
    for category, cat_results in sorted(results_by_category.items()):
        report.append(f"### {category}")
        report.append(f"Evaluations: {len(cat_results)}")
        report.append("")

    # Detailed conversation flows
    report.append("## Detailed Evaluation Results")

    # Build a map of session_id to eval_dataset for easy lookup
    session_to_eval_data = {}
    for eval_dataset in eval_datasets:
        for session in eval_dataset:
            session_to_eval_data[session.get('sessionId')] = session

    for idx, result in enumerate(results, 1):
        session_id = result.get('session_id')
        category = result.get('category', 'Unknown')

        report.append(f"### Evaluation {idx}")
        report.append(f"**Session ID:** {session_id}")
        report.append(f"**Category:** {category}")
        report.append(f"**Model:** {result.get('model_id', 'Unknown')}")
        report.append(f"**Tokens:** {result.get('total_tokens', 0)}")
        report.append(f"**Cost:** ${result.get('cost', 0.0):.6f}")
        report.append("")

        # Add evaluation criteria scores
        if 'criteria' in result:
            report.append("#### Evaluation Scores")
            for criterion, data in sorted(result.get('criteria', {}).items()):
                if isinstance(data, dict):
                    score = data.get('score', 'N/A')
                    explanation = data.get('explanation', '')
                    report.append(f"- **{criterion}**: {score}")
                    report.append(f"  - {explanation}")

        # Add overall assessment
        if 'overall' in result:
            report.append("")
            report.append("#### Overall Assessment")
            report.append(f"**Score:** {result['overall'].get('score', 'N/A')}")
            report.append(f"**Summary:** {result['overall'].get('summary', '')}")

        # Add conversation flow from eval_data
        eval_session = session_to_eval_data.get(session_id)
        if eval_session:
            turns = eval_session.get('turns', [])
            if turns:
                report.append("")
                report.append("#### Conversation Flow")

                for turn_idx, turn in enumerate(turns, 1):
                    report.append(f"##### Turn {turn_idx}")

                    if 'systemPrompt' in turn:
                        report.append(f"**System Prompt:** {turn['systemPrompt']}")

                    if 'user' in turn:
                        report.append(f"**User:** {turn['user']}")

                    if 'tools' in turn and turn['tools']:
                        report.append("**Tools Called:**")
                        for tool in turn['tools']:
                            tool_name = tool.get('tool_name', 'Unknown')
                            tool_result = tool.get('result', '')
                            report.append(f"- {tool_name}: {tool_result}")

                    if 'assistant' in turn:
                        report.append(f"**Assistant:** {turn['assistant']}")

                    report.append("")

        report.append("")

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Evaluate S2S interactions using LLM-as-Judge")
    parser.add_argument("--num-runs", type=int, default=1,
                       help="Number of evaluation runs (default: 1)")
    parser.add_argument("--delay", type=int, default=60,
                       help="Delay between runs in seconds (default: 60)")
    parser.add_argument("--llm-judge-config", default=DEFAULT_CONFIG_PATH,
                       help=f"Config file path (default: {DEFAULT_CONFIG_PATH})")
    parser.add_argument("--validation-dataset", default=DEFAULT_VALIDATION_DATASET_PATH,
                       help=f"Validation dataset path (default: {DEFAULT_VALIDATION_DATASET_PATH})")
    parser.add_argument("--eval-dataset", default=DEFAULT_EVAL_DATASET_PATH,
                       help=f"Eval dataset path (default: {DEFAULT_EVAL_DATASET_PATH})")
    parser.add_argument("--mappings-file", default=DEFAULT_MAPPINGS_PATH,
                       help=f"Manual mappings file path (default: {DEFAULT_MAPPINGS_PATH})")
    parser.add_argument("--category", default=None,
                       help="Filter by category")
    parser.add_argument("--report_output_path", default=DEFAULT_REPORT_OUTPUT_PATH,
                       help=f"Output path (default: {DEFAULT_REPORT_OUTPUT_PATH})")
    parser.add_argument("--session-ids", type=str, nargs='*', default=None,
                       help="Optional: Session IDs to extract from CloudWatch (space-separated)")
    parser.add_argument("--hours-back", type=int, default=24,
                       help="Hours back from now to query CloudWatch (default: 24)")
    parser.add_argument("--cloudwatch-region", default="us-east-1",
                       help="AWS region for CloudWatch (default: us-east-1)")
    parser.add_argument("--cloudwatch-log-group", default="aws/spans",
                       help="CloudWatch log group name (default: aws/spans)")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Enable debug logging for troubleshooting")

    args = parser.parse_args()

    # Enable debug logging if verbose flag is set
    if args.verbose:
        set_debug_logging(enabled=True)
        logger.info("Debug logging enabled")

    # Load config and datasets
    config = load_config(args.llm_judge_config)
    validation_data = load_validation_dataset(args.validation_dataset)
    eval_data = load_eval_dataset(args.eval_dataset)

    # Check if eval dataset exists and ask user what to do
    eval_dataset_exists = os.path.exists(args.eval_dataset) and len(eval_data) > 0

    if eval_dataset_exists:
        logger.info(f"Found existing eval dataset with {len(eval_data)} sessions")
        user_choice = input(
            "\nExisting eval dataset found. What would you like to do?\n"
            "1. Continue with existing eval data\n"
            "2. Overwrite and extract new traces from CloudWatch\n"
            "Enter choice (1 or 2): "
        ).strip()

        if user_choice == '2':
            logger.info("User chose to overwrite eval dataset and extract new traces")
            eval_data = []  # Clear to trigger extraction
        elif user_choice == '1':
            logger.info("User chose to continue with existing eval data")
        else:
            logger.error("Invalid choice. Please enter 1 or 2.")
            return 1

    # Generate eval dataset from CloudWatch if not provided
    if not eval_data:
        logger.info("Extracting traces from CloudWatch...")

        raw_traces = []

        if args.session_ids and len(args.session_ids) > 0:
            # Extract specific sessions if provided
            logger.info(f"Extracting traces for {len(args.session_ids)} specific session(s)")
            for session_id in args.session_ids:
                try:
                    logger.info(f"Extracting traces for session: {session_id}")
                    traces = extract_traces_from_cloudwatch(
                        session_id=session_id,
                        hours_back=args.hours_back,
                        region_name=args.cloudwatch_region,
                        log_group_name=args.cloudwatch_log_group
                    )
                    raw_traces.append(traces)
                    logger.info(f"Successfully extracted {traces.get('total_spans', 0)} spans for session {session_id}")
                except Exception as e:
                    logger.error(f"Failed to extract traces for session {session_id}: {e}")
                    continue
        else:
            # Extract all sessions from last N hours
            logger.info(f"Extracting all sessions from the last {args.hours_back} hour(s)")
            try:
                traces = extract_traces_from_cloudwatch(
                    session_id=None,
                    hours_back=args.hours_back,
                    region_name=args.cloudwatch_region,
                    log_group_name=args.cloudwatch_log_group
                )
                raw_traces.append(traces)
                logger.info(f"Successfully extracted {traces.get('total_spans', 0)} spans from {traces.get('total_sessions', 0)} session(s)")
            except Exception as e:
                logger.error(f"Failed to extract traces from CloudWatch: {e}")
                return 1

        if not raw_traces:
            logger.error("Failed to extract any traces from CloudWatch")
            return 1

        # Process and store eval dataset
        try:
            logger.info("Processing extracted traces into eval dataset...")
            eval_data = process_and_store_eval_dataset(raw_traces, args.eval_dataset)
            logger.info(f"Successfully generated eval dataset with {len(eval_data)} sessions")
        except Exception as e:
            logger.error(f"Failed to process and store eval dataset: {e}")
            return 1

    # Validate that we have evaluation data before proceeding
    if not eval_data:
        logger.error("No evaluation data available. Cannot proceed with evaluation runs.")
        logger.error("Ensure CloudWatch contains valid traces with session IDs or provide an existing eval dataset.")
        return 1

    # Load manual mappings (REQUIRED)
    manual_mappings = []
    if os.path.exists(args.mappings_file):
        try:
            with open(args.mappings_file, 'r') as f:
                manual_mappings = json.load(f)
            logger.info(f"✅ Loaded {len(manual_mappings)} manual mappings from {args.mappings_file}")
        except Exception as e:
            logger.error(f"Failed to load manual mappings from {args.mappings_file}: {e}")
            return 1
    else:
        logger.error(f"❌ Manual mappings file not found: {args.mappings_file}")
        logger.error("Manual mappings are REQUIRED to match CloudWatch sessions to validation categories")
        logger.error("Create mappings using the Jupyter notebook's Annotation UI (Cell 21)")
        logger.error(f"Or specify a different path with --mappings-file")
        return 1

    if not manual_mappings or len(manual_mappings) == 0:
        logger.error("❌ Manual mappings file is empty")
        logger.error("Manual mappings are REQUIRED to match CloudWatch sessions to validation categories")
        logger.error("Create mappings using the Jupyter notebook's Annotation UI (Cell 21)")
        return 1

    logger.info(f"Found {len(eval_data)} sessions in eval dataset")
    logger.info(f"Found {len(manual_mappings)} manual mappings")
    logger.info(f"Found {len(validation_data)} validation category templates")

    # Initialize judge
    judge = LLMJudge(config)

    # Generate a single timestamp for all outputs in this evaluation run
    evaluation_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    evaluation_output_dir = os.path.join(
        args.report_output_path,
        f"evaluation_report_{evaluation_timestamp}"
    )
    os.makedirs(evaluation_output_dir, exist_ok=True)
    logger.info(f"Created evaluation output directory: {evaluation_output_dir}")

    # Run evaluations
    run_results = []
    for run in range(args.num_runs):
        logger.info(f"Starting evaluation run {run + 1}/{args.num_runs}")

        iteration_result = run_evaluation_iteration(
            eval_data, validation_data, judge, args.category, manual_mappings
        )
        run_results.append(iteration_result)

        # Save run result
        run_file = os.path.join(evaluation_output_dir,
                               f"run_{run + 1}_results.json")
        with open(run_file, 'w') as f:
            json.dump(iteration_result, f, indent=2)
        logger.info(f"Saved run {run + 1} results to {run_file}")

        if run < args.num_runs - 1:
            logger.info(f"Waiting {args.delay} seconds before next run...")
            time.sleep(args.delay)

    # Merge results
    merged_results = merge_results(run_results)

    # Check if any evaluations were performed
    total_evaluations = merged_results.get('total_evaluations', 0)
    if total_evaluations == 0:
        logger.warning("No sessions were evaluated")
        logger.warning("Possible reasons:")
        logger.warning("  - Manual mappings reference sessions that don't exist in CloudWatch data")
        logger.warning("  - Mapped categories don't exist in validation dataset")
        logger.warning("  - All sessions were filtered out by category filter")
        logger.warning("Skipping report generation")
        return 1

    # Save merged results
    merged_file = os.path.join(evaluation_output_dir, "complete_results.json")
    with open(merged_file, 'w') as f:
        json.dump(merged_results, f, indent=2)
    logger.info(f"Saved merged results to {merged_file}")

    # Generate report
    report = generate_evaluation_report(merged_results)
    report_file = os.path.join(evaluation_output_dir, "evaluation_report.md")
    with open(report_file, 'w') as f:
        f.write(report)
    logger.info(f"Saved report to {report_file}")

    logger.info("Evaluation complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
