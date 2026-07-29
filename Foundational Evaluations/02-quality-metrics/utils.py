"""
Utility functions for US Cities Demographics Model Evaluation
Supports programmatic testing and LLM-as-a-Judge evaluation workflows
"""

import boto3
import os
import sys
import pandas as pd
import json
import re
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any
import random
import numpy as np
import matplotlib.pyplot as plt

# Model IDs are centralised in ../model_config.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model_config import DEFAULT_MODEL_ID, JUDGE_MODEL_ID

# Initialize Bedrock client
bedrock = boto3.client("bedrock-runtime")

# Judge prompt template for LLM-as-a-Judge evaluation
# Binary pass/fail verdict only — rating scales (1-5, 1-10) introduce implicit
# variation that hides actual failure modes. See 03_Evaluating_your_Judge.ipynb.
JUDGE_PROMPT_TEMPLATE = """
You will be given a question about US cities demographics and population data. 
Your task is to evaluate a model's response and deliver a binary pass/fail verdict.

Here is the question about US cities:
<question>{QUESTION}</question>

Here is the model's response:
<model_response>{MODEL_RESPONSE}</model_response>

Here is the context from the data:
<dataset>{context}</dataset>

**Dataset Context:** The response should be based on the US Cities Population Dataset containing 314 most populous US cities with the following features:
- **city**: City name
- **state**: Two-letter state abbreviation  
- **population**: Population count (may include commas/formatting)
- **land_area_mi2**: Land area in square miles
- **Coverage**: Cities from 8.4M+ (NYC) down to ~100K residents

First, analyze the question type and evaluate the model response based on:

1. **Data Accuracy**: Are population figures, city names, and geographic information correct?
2. **Calculation Correctness**: If calculations are involved (density, rankings, comparisons), are they mathematically sound?
3. **Geographic Knowledge**: Does the response demonstrate proper understanding of US geography and state locations?
4. **Analytical Depth**: For complex queries, does the response provide meaningful insights beyond basic data retrieval?
5. **Data Handling**: Does the response appropriately handle data formatting issues (commas in numbers, footnotes, etc.)?

## Verdict Criteria (binary — do NOT use a rating scale)
- PASS: The numerical data in the response matches the context. Approximate values (e.g., "about 2.4 million" for 2,390,125) are acceptable. Calculations are mathematically sound. Additional commentary that does not contradict the data is fine.
- FAIL: The response contains a specific but wrong number, an incorrect calculation, reverses a comparison, claims data is unavailable when it is in the context, or does not answer the question asked.

Then, classify the question type:
1. **Factual Lookup**: Simple data retrieval (population of specific city)
2. **Ranking/Comparison**: Ordering cities by metrics or comparing multiple cities
3. **Calculation-Based**: Requires mathematical operations (density, growth rates, etc.)
4. **Geographic Analysis**: Regional patterns, state-level analysis, geographic distribution
5. **Trend Analysis**: Population patterns, urban development insights

Provide your evaluation in the following format:

<analysis>
[Your detailed analysis of the response quality, noting any factual errors, missing information, or analytical strengths/weaknesses]
</analysis>

<question_type>factual_lookup/ranking_comparison/calculation_based/geographic_analysis/trend_analysis</question_type>

<complexity>Basic/Intermediate/Advanced</complexity>

<verdict>pass/fail</verdict>

<reasoning>
[Explanation for the verdict based on accuracy, completeness, analytical quality, and appropriate handling of the dataset characteristics]
</reasoning>

<improvements>
[Specific suggestions for how the response could be enhanced, if applicable]
</improvements>
"""

# Bedrock API Communication Functions
def bedrock_call(prompt: str) -> Dict[str, Any]:
    """Make a Bedrock call using Converse API with structured JSON response."""
    
    structured_prompt = f"""
    You will be asked questions about city populations and land areas.
    
    Answer the following question: {prompt}
    
    For direct questions about population, respond in this JSON format only:
    {{
        "answer": [numerical answer only, no commas or text],
        "city": [city name],
        "metric": "population"
    }}

    For direct questions about land area, respond in this JSON format only:
    {{
        "answer": [numerical answer as decimal, like 46.9],
        "city": [city name],
        "metric": "land_area_mi2"
    }}

    For comparison questions, respond in this JSON format only:
    {{
        "answer": [numerical answer for larger city],
        "city": [name of larger city],
        "metric": [what was compared],
        "comparison": true
    }}

    Respond with the JSON only, no additional text.
    """
    
    response = bedrock.converse(
        modelId=DEFAULT_MODEL_ID,
        messages=[
            {
                'role': 'user',
                'content': [
                    {
                        'text': structured_prompt
                    }
                ]
            }
        ],
        inferenceConfig={
            'maxTokens': 300
        }
    )
    
    response_text = response['output']['message']['content'][0]['text']
    
    # Strip markdown code fences if present (e.g. ```json ... ```)
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        # Remove opening fence (with optional language tag) and closing fence
        lines = cleaned.split("\n")
        lines = lines[1:]  # drop opening ```json
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    
    return json.loads(cleaned)

def generate_model_response(question: str, context_data: str = "") -> str:
    """Generate a model response to a cities question using Bedrock."""
    
    prompt = f"""
    You are an AI assistant with knowledge about US cities demographics. Answer the following question about US cities based on your knowledge.
    
    Question: {question}
    
    {f"Context data from dataset: {context_data}" if context_data else ""}
    
    Provide a clear, informative response, tone neutral. If the question involves calculations (like population density), show your work.
    """
        
    try:
        response = bedrock.converse(
            modelId=DEFAULT_MODEL_ID,
            messages=[
                {
                    'role': 'user',
                    'content': [{'text': prompt}]
                }
            ],
            inferenceConfig={
                'maxTokens': 500
            }
        )
        
        return response['output']['message']['content'][0]['text']
    except Exception as e:
        return f"Error generating response: {str(e)}"

def call_judge_model(prompt: str, model_id: str = JUDGE_MODEL_ID) -> str:
    """Call the judge model to evaluate a response using boto3 directly."""
    try:
        response = bedrock.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 1000}
        )
        
        return response["output"]["message"]["content"][0]["text"]
    except Exception as e:
        return f"Error: {str(e)}"

# Data Processing and Verification Functions
def verify_answer(response: Dict[str, Any], df: pd.DataFrame, question: str) -> bool:
    """Verify if the answer matches our dataset."""
    try:
        city = response['city']
        metric = response['metric']
        
        # Handle comparison questions
        if response.get('comparison'):
            cities = question.split(':')[1].strip().split(' or ')
            city1, city2 = [c.strip() for c in cities]
            
            # Get values and handle both int and float
            val1_raw = df[df['city'].str.contains(city1, case=False)][metric].values[0]
            val2_raw = df[df['city'].str.contains(city2, case=False)][metric].values[0]
            
            if isinstance(val1_raw, str):
                val1 = float(val1_raw.replace(',', ''))
            else:
                val1 = float(val1_raw)
                
            if isinstance(val2_raw, str):
                val2 = float(val2_raw.replace(',', ''))
            else:
                val2 = float(val2_raw)
            
            expected_city = city1 if val1 > val2 else city2
            return city.lower() in expected_city.lower() or expected_city.lower() in city.lower()
        
        # Handle direct questions - improved city matching
        matching_rows = df[df['city'].str.contains(city, case=False, regex=False)]
        
        # If no match, try without brackets/footnotes
        if len(matching_rows) == 0:
            city_clean = city.split('[')[0].strip()
            matching_rows = df[df['city'].str.contains(city_clean, case=False, regex=False)]
        
        # If still no match, try the other way around
        if len(matching_rows) == 0:
            for idx, row in df.iterrows():
                dataset_city_clean = row['city'].split('[')[0].strip()
                if city.lower() in dataset_city_clean.lower() or dataset_city_clean.lower() in city.lower():
                    matching_rows = df.iloc[[idx]]
                    break
        
        if len(matching_rows) == 0:
            print(f"No match found for city: '{city}'")
            return False
            
        actual_value = matching_rows[metric].values[0]
        
        # Handle population (integer) vs land_area (float)
        if metric == 'population':
            if isinstance(actual_value, str):
                actual_value = int(actual_value.replace(',', ''))
            answer = int(response['answer'])
            return answer == actual_value
        else:  # land_area_mi2
            if isinstance(actual_value, str):
                actual_value = float(actual_value.replace(',', ''))
            answer = float(response['answer'])
            return abs(answer - actual_value) < 0.1
            
    except Exception as e:
        print(f"Verification error: {str(e)}")
        return False

def extract_evaluation_components(evaluation_text: str) -> Dict:
    """Extract structured components from judge evaluation."""
    
    patterns = {
        'analysis': r'<analysis>(.*?)</analysis>',
        'question_type': r'<question_type>(.*?)</question_type>',
        'complexity': r'<complexity>(.*?)</complexity>',
        'verdict': r'<verdict>(.*?)</verdict>',
        'reasoning': r'<reasoning>(.*?)</reasoning>',
        'improvements': r'<improvements>(.*?)</improvements>'
    }
    
    extracted = {}
    
    for key, pattern in patterns.items():
        match = re.search(pattern, evaluation_text, re.DOTALL | re.IGNORECASE)
        if match:
            extracted[key] = match.group(1).strip()
        else:
            extracted[key] = None
    
    # Normalize verdict to lowercase pass/fail
    if extracted['verdict']:
        verdict = extracted['verdict'].strip().lower()
        extracted['verdict'] = verdict if verdict in ('pass', 'fail') else None
    
    # Boolean convenience column for aggregation
    extracted['passed'] = extracted['verdict'] == 'pass' if extracted['verdict'] else None
    
    return extracted

# Evaluation and Testing Framework
def run_tests(questions: List[str], df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Run all test questions and collect results."""
    results = []
    
    for i, question in enumerate(questions):
        print(f"Testing {i+1}/{len(questions)}: {question}")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = bedrock_call(question)
                is_correct = verify_answer(response, df, question)
                
                results.append({
                    "question": question,
                    "response": response,
                    "passed": is_correct
                })
                break
                
            except Exception as e:
                if "ThrottlingException" in str(e) and attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"   Throttled, waiting {wait_time:.1f}s before retry...")
                    sleep(wait_time)
                else:
                    print(f"   Error: {str(e)}")
                    results.append({
                        "question": question,
                        "error": str(e),
                        "passed": False
                    })
                    break
        
        sleep(2)
    
    return results

def call_threaded_evaluation(prompts: List[str], max_workers=3) -> List[str]:
    """Process evaluation requests in parallel using boto3."""
    future_to_position = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, prompt in enumerate(prompts):
            future = executor.submit(call_judge_model, prompt)
            future_to_position[future] = i
        
        responses = [None] * len(prompts)
        
        for future in as_completed(future_to_position):
            position = future_to_position[future]
            try:
                response = future.result()
                responses[position] = response
            except Exception as exc:
                print(f"Request at position {position} generated an exception: {exc}")
                responses[position] = f"Error: {str(exc)}"
        
    return responses

# Prompt Building and Template Management
def build_judge_prompt(question: str, model_response: str, context: str = "") -> str:
    """Build the judge prompt for evaluating US cities demographic analysis responses."""
    
    formatted_prompt = JUDGE_PROMPT_TEMPLATE.replace("{QUESTION}", question)
    formatted_prompt = formatted_prompt.replace("{MODEL_RESPONSE}", model_response)
    formatted_prompt = formatted_prompt.replace("{context}", context)
    
    return formatted_prompt

# Utility Functions for Results Processing
def save_evaluation_results(parsed_evaluations: List[Dict], output_prefix: str = "cities_evaluation"):
    """Save evaluation results to JSON and CSV files."""
    
    # Save detailed results to JSON
    json_file = f"{output_prefix}_results.json"
    with open(json_file, 'w') as f:
        json.dump(parsed_evaluations, f, indent=2, default=str)
    
    # Save summary CSV
    if parsed_evaluations:
        df_evaluations = pd.DataFrame(parsed_evaluations)
        summary_columns = ['question', 'verdict', 'question_type', 'complexity', 
                          'analysis', 'reasoning', 'improvements']
        
        available_columns = [col for col in summary_columns if col in df_evaluations.columns]
        if available_columns:
            summary_df = df_evaluations[available_columns]
            csv_file = f"{output_prefix}_summary.csv"
            summary_df.to_csv(csv_file, index=False)
            
            return json_file, csv_file
    
    return json_file, None

def calculate_evaluation_metrics(df_evaluations: pd.DataFrame) -> Dict[str, Any]:
    """Calculate summary metrics from evaluation results (pass/fail verdicts)."""
    
    if df_evaluations.empty or 'passed' not in df_evaluations.columns:
        return {}
    
    n_pass = int(df_evaluations['passed'].sum())
    total = len(df_evaluations)
    
    metrics = {
        'pass_count': n_pass,
        'fail_count': total - n_pass,
        'pass_rate': n_pass / total if total > 0 else 0,
        'total_evaluations': total
    }
    
    # Question type performance
    if 'question_type' in df_evaluations.columns:
        type_stats = df_evaluations.groupby('question_type')['passed'].agg(['mean', 'count'])
        type_stats.columns = ['pass_rate', 'count']
        metrics['question_type_performance'] = type_stats.to_dict('index')
    
    # Complexity performance
    if 'complexity' in df_evaluations.columns:
        complexity_stats = df_evaluations.groupby('complexity')['passed'].agg(['mean', 'count'])
        complexity_stats.columns = ['pass_rate', 'count']
        metrics['complexity_performance'] = complexity_stats.to_dict('index')
    
    return metrics



def generate_realistic_performance_data(n_samples=1000, random_seed=42):
    """
    Generate realistic judge verdict data that models degradation from perfect lab
    conditions to production reality across different question types.
    
    Verdicts are binary pass/fail — no rating scales.
    
    Args:
        n_samples (int): Number of evaluation samples to generate
        random_seed (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: DataFrame with 'verdict', 'passed' and 'question_type' columns
    """
    np.random.seed(random_seed)
    
    question_types = ['calculation_based', 'factual_lookup', 'ranking_comparison', 
                     'creative_writing', 'technical_explanation']
    
    # Realistic failure rates per question type in production
    fail_rates = {
        'ranking_comparison': 0.18,      # Most challenging - ambiguous criteria
        'creative_writing': 0.12,        # Subjective evaluation variance
        'factual_lookup': 0.05,          # Strong but some data quality issues
        'calculation_based': 0.03,       # Reliable, deterministic
        'technical_explanation': 0.03    # Reliable, well-structured
    }
    
    question_type_list = []
    verdicts = []
    
    for i in range(n_samples):
        q_type = np.random.choice(question_types)
        question_type_list.append(q_type)
        verdicts.append('fail' if np.random.random() < fail_rates[q_type] else 'pass')
    
    df = pd.DataFrame({
        'verdict': verdicts,
        'question_type': question_type_list
    })
    df['passed'] = df['verdict'] == 'pass'
    return df

def calculate_performance_stats(df):
    """
    Calculate comprehensive pass/fail statistics and identify failure clusters.
    
    Args:
        df (pd.DataFrame): DataFrame with 'verdict', 'passed' and 'question_type' columns
        
    Returns:
        dict: Dictionary containing all calculated statistics
    """
    total = len(df)
    n_pass = int(df['passed'].sum())
    pass_rate = n_pass / total if total > 0 else 0
    failures = df[~df['passed']]
    
    # Question type statistics
    question_stats = df.groupby('question_type').agg(
        pass_rate=('passed', 'mean'),
        count=('passed', 'count'),
        failures=('passed', lambda s: int((~s).sum()))
    ).round(3)
    
    # Failure counts per question type
    failure_counts = failures.groupby('question_type').size().reindex(question_stats.index, fill_value=0)
    
    # Detailed stats for each question type
    detailed_stats = []
    for q_type in question_stats.index:
        subset = df[df['question_type'] == q_type]
        fail_count = int((~subset['passed']).sum())
        fail_pct = (fail_count / len(subset)) * 100
        
        detailed_stats.append({
            'type': q_type,
            'pass_rate': subset['passed'].mean(),
            'failures': fail_count,
            'fail_pct': fail_pct
        })
    
    detailed_stats.sort(key=lambda x: x['fail_pct'], reverse=True)
    
    # Verdict distribution [pass, fail]
    verdict_counts = [n_pass, total - n_pass]
    
    return {
        'pass_rate': pass_rate,
        'n_pass': n_pass,
        'n_fail': total - n_pass,
        'failures': failures,
        'question_stats': question_stats,
        'failure_counts': failure_counts,
        'detailed_stats': detailed_stats,
        'verdict_counts': verdict_counts
    }

def create_performance_visualization(df, stats, figsize=(16, 10)):
    """
    Create comprehensive performance analysis visualization.
    
    Args:
        df (pd.DataFrame): Performance data
        stats (dict): Statistics from calculate_performance_stats()
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    plt.suptitle('Large-Scale Pass/Fail Analysis (N={:,})'.format(len(df)), 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 1. Verdict distribution
    _plot_verdict_distribution(axes[0,0], stats)
    
    # 2. Pass rate by question type
    _plot_pass_rate_by_type(axes[0,1], stats)
    
    # 3. Failure counts by question type
    _plot_failure_by_type(axes[0,2], stats)
    
    # 4. Rolling pass rate over evaluations
    _plot_rolling_pass_rate(axes[1,0], df, stats)
    
    # 5. Pass rate confidence intervals
    _plot_confidence_analysis(axes[1,1], df, stats)
    
    # 6. Production readiness summary
    _plot_production_summary(axes[1,2], stats)
    
    plt.tight_layout()
    return fig

def _plot_verdict_distribution(ax, stats):
    """Plot overall pass/fail verdict distribution."""
    colors = ['green', 'red']
    explode = (0, 0.1)
    
    ax.pie(stats['verdict_counts'], labels=['Pass', 'Fail'], 
           autopct='%1.1f%%', colors=colors, startangle=90, explode=explode)
    ax.set_title('Production Verdict Distribution', fontweight='bold')

def _plot_pass_rate_by_type(ax, stats):
    """Plot pass rate by question type."""
    x_pos = range(len(stats['question_stats']))
    pass_rates = stats['question_stats']['pass_rate'] * 100
    bars = ax.bar(x_pos, pass_rates, alpha=0.7,
                  color=['lightcoral' if count > 20 else 'steelblue' for count in stats['failure_counts']])
    
    ax.axhline(stats['pass_rate'] * 100, color='red', linestyle='-', linewidth=2, alpha=0.8, 
               label=f'Overall Pass Rate: {stats["pass_rate"]*100:.1f}%')
    ax.axhline(90, color='orange', linestyle='--', linewidth=2, alpha=0.8, label='Target: 90%')
    
    ax.set_title('Pass Rate by Question Type', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([qt.replace('_', '\n') for qt in stats['question_stats'].index], rotation=0, fontsize=9)
    ax.set_ylabel('Pass Rate (%)')
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    for i, (rate, fail_count) in enumerate(zip(pass_rates, stats['failure_counts'])):
        if fail_count > 0:
            ax.text(i, rate + 1.5, f'{fail_count} fails', 
                   ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')

def _plot_failure_by_type(ax, stats):
    """Plot failure counts per question type."""
    x_pos = range(len(stats['failure_counts']))
    counts = stats['failure_counts'].values
    ax.bar(x_pos, counts, alpha=0.7,
           color=['lightcoral' if c > 20 else 'steelblue' for c in counts])
    
    ax.set_title('Failures by Question Type', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([qt.replace('_', '\n') for qt in stats['failure_counts'].index], rotation=0, fontsize=9)
    ax.set_ylabel('Failure Count')
    ax.grid(True, alpha=0.3)
    
    for i, c in enumerate(counts):
        ax.text(i, c + 0.5, str(int(c)), ha='center', va='bottom', fontsize=9, fontweight='bold')

def _plot_rolling_pass_rate(ax, df, stats, window=50):
    """Plot rolling pass rate over evaluation sequence to check stability."""
    rolling = df['passed'].rolling(window=window, min_periods=window).mean() * 100
    ax.plot(df.index, rolling, color='steelblue', linewidth=2, 
            label=f'Rolling pass rate (window={window})')
    
    # Mark failures along the sequence
    failures = stats['failures']
    ax.scatter(failures.index, [0] * len(failures), color='red', s=15, marker='x', 
               alpha=0.5, label=f'Failures ({len(failures)})')
    
    ax.axhline(stats['pass_rate'] * 100, color='red', linestyle='-', linewidth=2, alpha=0.8, 
               label='Overall Pass Rate')
    ax.axhline(90, color='orange', linestyle='--', linewidth=2, alpha=0.8, label='Target: 90%')
    
    ax.set_title('Pass Rate Stability Over Time', fontweight='bold')
    ax.set_xlabel('Evaluation ID')
    ax.set_ylabel('Pass Rate (%)')
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

def _plot_confidence_analysis(ax, df, stats):
    """Plot per-type pass rates with 95% binomial confidence intervals."""
    types = list(stats['question_stats'].index)
    rates = []
    errors = []
    
    for q_type in types:
        subset = df[df['question_type'] == q_type]
        n = len(subset)
        p = subset['passed'].mean()
        # Normal approximation 95% CI for a proportion
        se = np.sqrt(p * (1 - p) / n) if n > 0 else 0
        rates.append(p * 100)
        errors.append(1.96 * se * 100)
    
    x_pos = range(len(types))
    ax.bar(x_pos, rates, yerr=errors, capsize=5, alpha=0.7,
           color=['lightcoral' if count > 20 else 'steelblue' for count in stats['failure_counts']])
    
    ax.axhline(stats['pass_rate'] * 100, color='red', linestyle='-', linewidth=2, alpha=0.8, 
               label='Overall Pass Rate')
    ax.set_title('Pass Rate 95% Confidence Intervals', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([qt.replace('_', '\n') for qt in types], rotation=0, fontsize=9)
    ax.set_ylabel('Pass Rate (%)')
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)

def _plot_production_summary(ax, stats):
    """Plot production readiness summary."""
    ax.axis('off')
    
    total = stats['n_pass'] + stats['n_fail']
    summary_text = f"""PRODUCTION READINESS ASSESSMENT:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Overall Pass Rate: {stats['pass_rate']*100:.1f}%
        Passed: {stats['n_pass']} / {total}
        Failed: {stats['n_fail']} cases ({stats['n_fail']/total*100:.1f}%)

        ✓ SLA EXPECTATION: {stats['pass_rate']*100:.1f}% passing
        ✓ CAPACITY PLANNING: {stats['n_fail']} cases need human review
        ✓ MONITORING: alert when pass rate < 90%
        ✓ IMPROVEMENT PRIORITY: {stats['detailed_stats'][0]['type'].replace('_', ' ').title()}

        STATISTICAL SIGNIFICANCE: Large sample 
        narrows pass-rate confidence intervals
        """
    
    ax.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=11, 
            family='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    ax.set_title('Production Deployment Readiness', fontweight='bold')

def print_performance_summary(stats):
    """Print performance analysis summary."""
    total_samples = stats['n_pass'] + stats['n_fail']
    fail_pct = stats['n_fail'] / total_samples * 100

    print("Large-scale analysis complete! Model shows realistic pass/fail distribution")
    print(f"Degradation from perfect lab conditions (100%) to production reality ({stats['pass_rate']*100:.1f}% pass rate)")
    print(f"{stats['n_fail']} cases ({fail_pct:.1f}%) failed and require human review - typical for scale")
    print(f"Worst-performing question type: {stats['detailed_stats'][0]['type'].replace('_', ' ')} ({stats['detailed_stats'][0]['fail_pct']:.1f}% failure rate)")

def load_and_explore_dataset(filepath="./city_pop.csv"):
    """
    Load the US Cities Population Dataset and perform initial exploration.

    Args:
        filepath (str): Path to the CSV file

    Returns:
        pd.DataFrame: Loaded dataset
    """
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} cities, {df.shape[1]} features")
    print(f"Features available: {list(df.columns)}")

    # Display sample data for context
    print("\nSample city records:")
    for i in range(min(3, len(df))):
        print(f"\n--- City {i+1} ---")
        print(f"City: {df.iloc[i]['city']}")
        print(f"State: {df.iloc[i]['state']}")
        print(f"Population: {df.iloc[i]['population']}")
        print(f"Land Area: {df.iloc[i]['land_area_mi2']} sq mi")

    return df

def run_programmatic_tests(test_questions, df):
    """
    Run programmatic tests against ground truth dataset.

    Args:
        test_questions (list): List of test questions
        df (pd.DataFrame): Ground truth dataset

    Returns:
        list: Test results with pass/fail status
    """
    print("Testing model responses against ground truth dataset")

    test_results = run_tests(test_questions, df)

    passed_tests = sum(1 for result in test_results if result['passed'])
    print(f"\nPROGRAMMATIC TEST RESULTS:")
    print(f"Passed: {passed_tests}/{len(test_questions)}")
    print(f"Success Rate: {(passed_tests/len(test_questions))*100:.2f}%")

    print("\n Detailed Results:")
    for result in test_results:
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        print(f"{status} - {result['question']}")
        if 'response' in result:
            print(f"   Response: {json.dumps(result['response'], indent=2)}")
        elif 'error' in result:
            print(f"   Error: {result['error']}")

    return test_results

def generate_contextual_responses(cities_questions):
    """
    Generate model responses with contextual information using RAG approach.

    Args:
        cities_questions (list): List of dictionaries with 'question' and 'context' keys

    Returns:
        list: List of responses with model outputs
    """
    print(" Generating contextual model responses...")
    print(" Using RAG approach with dataset context for grounding")

    cities_responses = []
    for i, item in enumerate(cities_questions):
        print(f"⚡ Processing {i+1}/{len(cities_questions)}: {item['question']}")

        # The utility function handles Bedrock API calls and response formatting
        model_response = generate_model_response(item['question'], item['context'])

        cities_responses.append({
            'question': item['question'],
            'model_response': model_response,
            'context': item['context']
        })

        sleep(1)  # Rate limiting

    print(f"\nGenerated {len(cities_responses)} contextual responses for evaluation")
    return cities_responses

def run_judge_evaluations(cities_responses):
    """
    Run LLM-as-a-Judge evaluations on model responses.

    Args:
        cities_responses (list): List of model responses to evaluate

    Returns:
        list: Evaluation results from judge model
    """
    print("Starting LLM-as-a-Judge evaluation...")

    # Build evaluation prompts using our structured template
    evaluation_prompts = []
    for response_data in cities_responses:
        judge_prompt = build_judge_prompt(
            question=response_data['question'],
            model_response=response_data['model_response'],
            context=response_data.get('context', '')
        )
        evaluation_prompts.append(judge_prompt)

    print(" Running parallel evaluations for efficiency..")

    # Process evaluations concurrently to save time
    evaluation_results = call_threaded_evaluation(evaluation_prompts)

    print(f" Completed {len(evaluation_results)} comprehensive evaluations")
    return evaluation_results

def process_evaluation_results(cities_responses, evaluation_results):
    """
    Process and analyze evaluation results to extract insights.

    Args:
        cities_responses (list): Original model responses
        evaluation_results (list): Judge evaluation results

    Returns:
        tuple: (parsed_evaluations, df_evaluations, metrics)
    """
    print(" Processing evaluation results and extracting insights.")

    # Parse structured evaluation components
    parsed_evaluations = []
    for i, (response_data, evaluation_text) in enumerate(zip(cities_responses, evaluation_results)):
        if not evaluation_text.startswith("Error:"):
            # Extract structured components (scores, reasoning, improvements)
            parsed_eval = extract_evaluation_components(evaluation_text)

            combined_result = {
                **response_data,
                'evaluation_text': evaluation_text,
                **parsed_eval
            }
            parsed_evaluations.append(combined_result)

    # Create analysis dataframe and calculate metrics
    df_evaluations = pd.DataFrame(parsed_evaluations)
    metrics = calculate_evaluation_metrics(df_evaluations)

    print("\nEvaluation DataFrame created with columns:")
    print(list(df_evaluations.columns))

    # Display summary statistics for cities evaluation
    if not df_evaluations.empty:
        n_pass = int(df_evaluations['passed'].sum())
        total = len(df_evaluations)
        print(f"\nEvaluation Summary:")
        print(f"Pass Rate: {n_pass/total*100:.1f}% ({n_pass}/{total} passed)")

        print(f"\nQuestion Type Distribution:")
        print(df_evaluations['question_type'].value_counts())

        print(f"\nComplexity Distribution:")
        print(df_evaluations['complexity'].value_counts())
    else:
        print("No evaluation results to parse.")

    return parsed_evaluations, df_evaluations, metrics

def display_evaluation_metrics(csv_file="cities_evaluation_summary.csv"):
    """
    Load and display evaluation metrics from CSV file.

    Args:
        csv_file (str): Path to the CSV file with evaluation metrics
    """
    import os

    if os.path.exists(csv_file):
        print(f"Loading metrics from: {csv_file}")

        # Load the CSV file
        metrics_df = pd.read_csv(csv_file)

        print(f"\nCITIES EVALUATION METRICS TABLE ({len(metrics_df)} records)")
        print("=" * 80)

        # Configure pandas display options for better viewing
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 50)

        # Display the dataframe
        print(metrics_df.to_string())

        # Display additional cities-specific analysis
        if not metrics_df.empty and 'verdict' in metrics_df.columns:
            print(f"\n" + "=" * 80)
            print("CITIES EVALUATION ANALYSIS")
            print("=" * 80)

            passed = metrics_df['verdict'].str.lower() == 'pass'
            print(f"\nVERDICT STATISTICS:")
            print(f"  Pass Rate: {passed.mean()*100:.1f}%")
            print(f"  Passed: {int(passed.sum())} | Failed: {int((~passed).sum())}")

            if 'question_type' in metrics_df.columns:
                print(f"\nQUESTION TYPE PERFORMANCE:")
                type_stats = passed.groupby(metrics_df['question_type']).agg(['mean', 'count'])
                for question_type, stats in type_stats.iterrows():
                    print(f"  {question_type.replace('_', ' ').title()}: {stats['mean']*100:.1f}% pass rate ({int(stats['count'])} questions)")

            if 'complexity' in metrics_df.columns:
                print(f"\nCOMPLEXITY PERFORMANCE:")
                complexity_stats = passed.groupby(metrics_df['complexity']).agg(['mean', 'count'])
                for complexity, stats in complexity_stats.iterrows():
                    print(f"  {complexity}: {stats['mean']*100:.1f}% pass rate ({int(stats['count'])} questions)")

        return metrics_df
    else:
        print(f"CSV file not found: {csv_file}")
        print("Please run the cities evaluation cells first to generate the metrics file.")
        print("\nExpected workflow:")
        print("1. Generate model responses to cities questions")
        print("2. Run LLM-as-a-Judge evaluation")
        print("3. Parse and save evaluation results")
        print("4. View this metrics summary")
        return None

def analyze_model_performance(n_samples=1000, random_seed=42, show_plots=False):
    """
    Complete model performance analysis pipeline.

    Args:
        n_samples (int): Number of samples to generate
        random_seed (int): Random seed for reproducibility
        show_plots (bool): Whether to display the visualization

    Returns:
        tuple: (DataFrame, stats_dict, matplotlib.figure.Figure or None)
    """
    # Generate data
    df = generate_realistic_performance_data(n_samples, random_seed)

    # Calculate statistics
    stats = calculate_performance_stats(df)

    # Print basic stats
    print(f"Overall Pass Rate: {stats['pass_rate']*100:.1f}% ({stats['n_pass']}/{stats['n_pass'] + stats['n_fail']})")
    print(f"Failures: {stats['n_fail']} cases")
    print("Failure breakdown by question type:")
    print(stats['failures'].groupby('question_type').size().sort_values(ascending=False))

    # Only create visualization if requested
    fig = None
    if show_plots:
        fig = create_performance_visualization(df, stats)
        plt.show()

    # Print summary
    print_performance_summary(stats)

    return df, stats, fig

def create_evaluation_summary(test_results, df_evaluations=None):
    """
    Create a comprehensive evaluation summary with visual indicators.

    Args:
        test_results (list): Results from programmatic testing
        df_evaluations (pd.DataFrame): Results from LLM judge evaluation

    Returns:
        dict: Summary statistics and insights
    """
    summary = {}

    # Programmatic test summary
    if test_results:
        passed = sum(1 for r in test_results if r.get('passed', False))
        total = len(test_results)
        summary['programmatic'] = {
            'passed': passed,
            'failed': total - passed,
            'total': total,
            'success_rate': (passed/total)*100 if total > 0 else 0
        }

    # Judge evaluation summary (binary pass/fail verdicts)
    if df_evaluations is not None and not df_evaluations.empty:
        n_pass = int(df_evaluations['passed'].sum())
        total = len(df_evaluations)
        summary['judge'] = {
            'passed': n_pass,
            'failed': total - n_pass,
            'pass_rate': (n_pass / total) * 100 if total > 0 else 0,
            'total_evaluated': total
        }

        # Question type breakdown
        if 'question_type' in df_evaluations.columns:
            type_performance = df_evaluations.groupby('question_type')['passed'].agg(['mean', 'count'])
            summary['by_type'] = type_performance.to_dict('index')

    return summary

def print_evaluation_dashboard(summary):
    """
    Print a formatted dashboard of evaluation results.

    Args:
        summary (dict): Summary statistics from create_evaluation_summary
    """
    print("\n" + "="*80)
    print(" " * 25 + "EVALUATION DASHBOARD 📊")
    print("="*80)

    # Programmatic Testing Results
    if 'programmatic' in summary:
        prog = summary['programmatic']
        print("\nPROGRAMMATIC TESTING")
        print("-" * 40)

        # Visual pass/fail bar
        passed_pct = prog['success_rate']
        bar_length = 30
        filled = int(bar_length * passed_pct / 100)
        bar = "█" * filled + "░" * (bar_length - filled)

        print(f"Success Rate: [{bar}] {passed_pct:.1f}%")
        print(f"Results: ✅ {prog['passed']} passed | ❌ {prog['failed']} failed | Total: {prog['total']}")

    # Judge Evaluation Results
    if 'judge' in summary:
        judge = summary['judge']
        print("\nLLM JUDGE EVALUATION")
        print("-" * 40)

        # Pass rate visualization
        pass_rate = judge['pass_rate']
        bar_length = 20
        filled = int(bar_length * pass_rate / 100)
        pass_bar = "🟩" * filled + "⬜" * (bar_length - filled)

        print(f"Pass Rate: {pass_bar} {pass_rate:.1f}%")
        print(f"Results: ✅ {judge['passed']} passed | ❌ {judge['failed']} failed")
        print(f"Total Evaluated: {judge['total_evaluated']}")

    # Question Type Performance
    if 'by_type' in summary:
        print("\nPERFORMANCE BY QUESTION TYPE")
        print("-" * 40)
        for q_type, stats in summary['by_type'].items():
            # Mini bar for each type (pass rate)
            mini_bar_length = 10
            mini_filled = int(mini_bar_length * stats['mean'])
            mini_bar = "▰" * mini_filled + "▱" * (mini_bar_length - mini_filled)

            print(f"{q_type.replace('_', ' ').title():25} {mini_bar} {stats['mean']*100:.0f}% pass ({int(stats['count'])} samples)")

    print("\n" + "="*80)

def create_quick_experiment(df, sample_size=3):
    """
    Create a quick experiment with sample questions for interactive testing.

    Args:
        df (pd.DataFrame): Dataset to sample from
        sample_size (int): Number of sample questions to generate

    Returns:
        list: Sample questions for testing
    """
    # Sample random cities
    sample_cities = df.sample(min(sample_size, len(df)))

    questions = []
    for _, city in sample_cities.iterrows():
        # Create different question types
        question_types = [
            f"What is the population of {city['city'].split('[')[0]}?",
            f"What is the land area of {city['city'].split('[')[0]} in square miles?",
            f"Calculate the population density of {city['city'].split('[')[0]}."
        ]
        questions.append(np.random.choice(question_types))

    return questions

def validate_environment():
    """
    Validate that all required dependencies and configurations are present.

    Returns:
        dict: Validation results with status and messages
    """
    validation = {
        'status': 'ready',
        'checks': {},
        'messages': []
    }

    # Check imports
    try:
        import boto3
        validation['checks']['boto3'] = '✅ Available'
    except ImportError:
        validation['checks']['boto3'] = '❌ Missing'
        validation['messages'].append("Install boto3: pip install boto3")
        validation['status'] = 'not_ready'

    try:
        import pandas
        validation['checks']['pandas'] = '✅ Available'
    except ImportError:
        validation['checks']['pandas'] = '❌ Missing'
        validation['messages'].append("Install pandas: pip install pandas")
        validation['status'] = 'not_ready'

    try:
        import matplotlib
        validation['checks']['matplotlib'] = '✅ Available'
    except ImportError:
        validation['checks']['matplotlib'] = '❌ Missing'
        validation['messages'].append("Install matplotlib: pip install matplotlib")
        validation['status'] = 'not_ready'

    # Check AWS credentials
    try:
        import boto3
        session = boto3.Session()
        credentials = session.get_credentials()
        if credentials:
            validation['checks']['aws_credentials'] = '✅ Configured'
        else:
            validation['checks']['aws_credentials'] = '⚠️ Not configured'
            validation['messages'].append("Configure AWS credentials for Bedrock access")
    except Exception:
        validation['checks']['aws_credentials'] = '⚠️ Unable to verify'

    return validation



def create_single_plot(plot_type, df, stats, figsize=(8, 6)):
    """Create and display a single plot type."""
    plt.figure(figsize=figsize)
    
    if plot_type == "verdict_distribution":
        _plot_verdict_distribution(plt.gca(), stats)
    elif plot_type == "pass_rate_by_type":
        _plot_pass_rate_by_type(plt.gca(), stats)
    elif plot_type == "failure_by_type":
        _plot_failure_by_type(plt.gca(), stats)
    elif plot_type == "rolling_pass_rate":
        _plot_rolling_pass_rate(plt.gca(), df, stats)
    elif plot_type == "confidence_analysis":
        _plot_confidence_analysis(plt.gca(), df, stats)
    elif plot_type == "production_summary":
        _plot_production_summary(plt.gca(), stats)
    
    plt.tight_layout()
    plt.show()

def display_plot_with_analysis(plot_type, df, stats, analysis_text, figsize=(8, 6)):
    """Display a single plot followed by analysis text."""
    create_single_plot(plot_type, df, stats, figsize)
    print(analysis_text)




def format_progress_bar(current, total, width=50):
    """
    Create a text-based progress bar.

    Args:
        current (int): Current progress
        total (int): Total items
        width (int): Width of the progress bar

    Returns:
        str: Formatted progress bar string
    """
    if total == 0:
        return "[" + "=" * width + "] Complete"

    progress = current / total
    filled = int(width * progress)
    bar = "=" * filled + ">" + "-" * (width - filled - 1)
    percentage = progress * 100

    return f"[{bar}] {percentage:.1f}% ({current}/{total})"