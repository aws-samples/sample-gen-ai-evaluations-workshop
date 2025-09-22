"""
Utility functions for US Cities Demographics Model Evaluation
Supports programmatic testing and LLM-as-a-Judge evaluation workflows
"""

import boto3
import pandas as pd
import json
import re
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any
import random
import numpy as np
import matplotlib.pyplot as plt

# Initialize Bedrock client
bedrock = boto3.client("bedrock-runtime")

# Judge prompt template for LLM-as-a-Judge evaluation
JUDGE_PROMPT_TEMPLATE = """
You will be given a question about US cities demographics and population data. 
Your task is to evaluate a model's response for accuracy, completeness, and analytical quality.

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

<score>X/10</score>

<reasoning>
[Explanation for the score based on accuracy, completeness, analytical quality, and appropriate handling of the dataset characteristics]
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
        modelId='us.anthropic.claude-sonnet-4-20250514-v1:0',
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
            'maxTokens': 300,
            'temperature': 0
        }
    )
    
    response_text = response['output']['message']['content'][0]['text']
    return json.loads(response_text)

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
            modelId='us.anthropic.claude-3-7-sonnet-20250219-v1:0',
            messages=[
                {
                    'role': 'user',
                    'content': [{'text': prompt}]
                }
            ],
            inferenceConfig={
                'maxTokens': 500,
                'temperature': 0.1
            }
        )
        
        return response['output']['message']['content'][0]['text']
    except Exception as e:
        return f"Error generating response: {str(e)}"

def call_judge_model(prompt: str, model_id: str = "us.anthropic.claude-3-7-sonnet-20250219-v1:0") -> str:
    """Call the judge model to evaluate a response using boto3 directly."""
    try:
        response = bedrock.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"temperature": 0.1, "maxTokens": 1000}
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
        'score': r'<score>(.*?)</score>',
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
    
    # Extract numeric score
    if extracted['score']:
        score_match = re.search(r'(\d+(?:\.\d+)?)', extracted['score'])
        if score_match:
            try:
                extracted['numeric_score'] = float(score_match.group(1))
            except ValueError:
                extracted['numeric_score'] = None
        else:
            extracted['numeric_score'] = None
    else:
        extracted['numeric_score'] = None
    
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
        summary_columns = ['question', 'numeric_score', 'question_type', 'complexity', 
                          'analysis', 'reasoning', 'improvements']
        
        available_columns = [col for col in summary_columns if col in df_evaluations.columns]
        if available_columns:
            summary_df = df_evaluations[available_columns]
            csv_file = f"{output_prefix}_summary.csv"
            summary_df.to_csv(csv_file, index=False)
            
            return json_file, csv_file
    
    return json_file, None

def calculate_evaluation_metrics(df_evaluations: pd.DataFrame) -> Dict[str, Any]:
    """Calculate summary metrics from evaluation results."""
    
    if df_evaluations.empty or 'numeric_score' not in df_evaluations.columns:
        return {}
    
    metrics = {
        'average_score': df_evaluations['numeric_score'].mean(),
        'median_score': df_evaluations['numeric_score'].median(),
        'min_score': df_evaluations['numeric_score'].min(),
        'max_score': df_evaluations['numeric_score'].max(),
        'total_evaluations': len(df_evaluations)
    }
    
    # Question type performance
    if 'question_type' in df_evaluations.columns:
        type_stats = df_evaluations.groupby('question_type')['numeric_score'].agg(['mean', 'count'])
        metrics['question_type_performance'] = type_stats.to_dict('index')
    
    # Complexity performance
    if 'complexity' in df_evaluations.columns:
        complexity_stats = df_evaluations.groupby('complexity')['numeric_score'].agg(['mean', 'count'])
        metrics['complexity_performance'] = complexity_stats.to_dict('index')
    
    return metrics



def generate_realistic_performance_data(n_samples=1000, random_seed=42):
    """
    Generate realistic performance data that models degradation from perfect lab conditions 
    to production reality across different question types.
    
    Args:
        n_samples (int): Number of evaluation samples to generate
        random_seed (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: DataFrame with 'numeric_score' and 'question_type' columns
    """
    np.random.seed(random_seed)
    
    realistic_scores = []
    question_types = ['calculation_based', 'factual_lookup', 'ranking_comparison', 
                     'creative_writing', 'technical_explanation']
    question_type_list = []

    for i in range(n_samples):
        q_type = np.random.choice(question_types)
        question_type_list.append(q_type)
        
        # Model realistic performance degradation from perfect lab conditions
        if q_type == 'ranking_comparison':
            # Most challenging - subjective evaluation, ambiguous criteria
            rand = np.random.random()
            if rand < 0.45:  # 45% excellent
                score = np.random.uniform(9.0, 10.0)
            elif rand < 0.80:  # 35% good
                score = np.random.uniform(7.5, 9.0)
            else:  # 20% needs work
                score = np.random.uniform(5.0, 7.5)
        elif q_type == 'creative_writing':
            # Subjective evaluation variance
            rand = np.random.random()
            if rand < 0.60:  # 60% excellent
                score = np.random.uniform(8.8, 10.0)
            elif rand < 0.88:  # 28% good
                score = np.random.uniform(7.0, 8.8)
            else:  # 12% problematic
                score = np.random.uniform(5.5, 7.0)
        elif q_type == 'factual_lookup':
            # Generally strong but some data quality issues
            rand = np.random.random()
            if rand < 0.75:  # 75% excellent
                score = np.random.uniform(9.2, 10.0)
            elif rand < 0.95:  # 20% good
                score = np.random.uniform(8.0, 9.2)
            else:  # 5% needs work
                score = np.random.uniform(6.0, 8.0)
        else:
            # Technical tasks remain most reliable
            rand = np.random.random()
            if rand < 0.82:  # 82% excellent
                score = np.random.uniform(9.3, 10.0)
            elif rand < 0.97:  # 15% good
                score = np.random.uniform(8.2, 9.3)
            else:  # 3% needs work
                score = np.random.uniform(6.5, 8.2)
        
        realistic_scores.append(round(score, 1))
    
    return pd.DataFrame({
        'numeric_score': realistic_scores,
        'question_type': question_type_list
    })

def calculate_performance_stats(df):
    """
    Calculate comprehensive performance statistics and identify outliers.
    
    Args:
        df (pd.DataFrame): DataFrame with 'numeric_score' and 'question_type' columns
        
    Returns:
        dict: Dictionary containing all calculated statistics
    """
    mean_score = df['numeric_score'].mean()
    std_score = df['numeric_score'].std()
    outlier_threshold = mean_score - 2 * std_score
    outliers = df[df['numeric_score'] < outlier_threshold]
    
    # Question type statistics
    question_stats = df.groupby('question_type').agg({
        'numeric_score': ['mean', 'std', 'count', 'min']
    }).round(2)
    question_stats.columns = ['mean', 'std', 'count', 'min']
    
    # Outlier counts per question type
    outlier_counts = outliers.groupby('question_type').size().reindex(question_stats.index, fill_value=0)
    
    # Detailed stats for each question type
    detailed_stats = []
    for q_type in question_stats.index:
        subset = df[df['question_type'] == q_type]
        outlier_count = len(outliers[outliers['question_type'] == q_type])
        outlier_pct = (outlier_count / len(subset)) * 100
        
        detailed_stats.append({
            'type': q_type,
            'mean': subset['numeric_score'].mean(),
            'std': subset['numeric_score'].std(),
            'outliers': outlier_count,
            'outlier_pct': outlier_pct
        })
    
    detailed_stats.sort(key=lambda x: x['outlier_pct'], reverse=True)
    
    # Quality distribution
    quality_counts = [
        len(df[df['numeric_score'] >= 9]),
        len(df[(df['numeric_score'] >= 7) & (df['numeric_score'] < 9)]),
        len(df[df['numeric_score'] < 7])
    ]
    
    return {
        'mean_score': mean_score,
        'std_score': std_score,
        'outlier_threshold': outlier_threshold,
        'outliers': outliers,
        'question_stats': question_stats,
        'outlier_counts': outlier_counts,
        'detailed_stats': detailed_stats,
        'quality_counts': quality_counts
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
    plt.suptitle('Large-Scale Performance Distribution Analysis (N={:,})'.format(len(df)), 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 1. Score distribution
    _plot_score_distribution(axes[0,0], df, stats)
    
    # 2. Performance by question type
    _plot_performance_by_type(axes[0,1], stats)
    
    # 3. Quality distribution pie chart
    _plot_quality_distribution(axes[0,2], stats)
    
    # 4. Outlier scatter plot
    _plot_outlier_analysis(axes[1,0], df, stats)
    
    # 5. Box plot variance analysis
    _plot_variance_analysis(axes[1,1], df, stats)
    
    # 6. Production readiness summary
    _plot_production_summary(axes[1,2], stats)
    
    plt.tight_layout()
    return fig

def _plot_score_distribution(ax, df, stats):
    """Plot score distribution with outlier highlighting."""
    ax.hist(df['numeric_score'], bins=25, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(stats['mean_score'], color='red', linestyle='-', linewidth=2, 
               label=f'Mean: {stats["mean_score"]:.2f}')
    ax.axvline(stats['outlier_threshold'], color='orange', linestyle='--', linewidth=2, 
               label=f'Outlier Threshold: {stats["outlier_threshold"]:.2f}')
    
    ax.axvspan(df['numeric_score'].min(), stats['outlier_threshold'], alpha=0.3, color='red', 
               label=f'Outliers: {len(stats["outliers"])} cases')
    
    ax.set_title('Production Performance Distribution', fontweight='bold')
    ax.set_xlabel('Score')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax.text(0.02, 0.98, f'Realistic degradation from\nperfect lab conditions\n({len(stats["outliers"])/len(df)*100:.1f}% require review)', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3), fontsize=10)

def _plot_performance_by_type(ax, stats):
    """Plot performance by question type."""
    x_pos = range(len(stats['question_stats']))
    bars = ax.bar(x_pos, stats['question_stats']['mean'], 
                  yerr=stats['question_stats']['std'], capsize=5, alpha=0.7,
                  color=['lightcoral' if count > 2 else 'steelblue' for count in stats['outlier_counts']])
    
    ax.axhline(stats['mean_score'], color='red', linestyle='-', linewidth=2, alpha=0.8, label='Overall Mean')
    ax.axhline(stats['outlier_threshold'], color='orange', linestyle='--', linewidth=2, alpha=0.8, label='Investigation Threshold')
    
    ax.set_title('Performance by Complexity & Context', fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([qt.replace('_', '\n') for qt in stats['question_stats'].index], rotation=0, fontsize=9)
    ax.set_ylabel('Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    for i, (mean_val, outlier_count) in enumerate(zip(stats['question_stats']['mean'], stats['outlier_counts'])):
        if outlier_count > 0:
            ax.text(i, mean_val + 0.5, f'{outlier_count} outliers', 
                   ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')

def _plot_quality_distribution(ax, stats):
    """Plot quality distribution pie chart."""
    colors = ['green', 'orange', 'red']
    explode = (0, 0, 0.1)
    
    ax.pie(stats['quality_counts'], labels=['Excellent (9-10)', 'Good (7-9)', 'Needs Investigation (<7)'], 
           autopct='%1.1f%%', colors=colors, startangle=90, explode=explode)
    ax.set_title('Statistical Significance Assessment', fontweight='bold')

def _plot_outlier_analysis(ax, df, stats):
    """Plot outlier analysis scatter plot."""
    colors_map = {'calculation_based': 'steelblue', 'factual_lookup': 'green', 
                  'ranking_comparison': 'red', 'creative_writing': 'orange', 
                  'technical_explanation': 'purple'}
    
    for q_type in df['question_type'].unique():
        subset = df[df['question_type'] == q_type]
        ax.scatter(subset.index, subset['numeric_score'], 
                  label=q_type.replace('_', ' ').title(), alpha=0.6, s=30,
                  color=colors_map.get(q_type, 'gray'))
    
    ax.scatter(stats['outliers'].index, stats['outliers']['numeric_score'], 
              color='red', s=100, marker='x', linewidth=3, label='Outliers')
    
    ax.axhline(stats['mean_score'], color='red', linestyle='-', linewidth=2, alpha=0.8, label='Mean')
    ax.axhline(stats['outlier_threshold'], color='orange', linestyle='--', linewidth=2, alpha=0.8, label='Threshold')
    
    ax.set_title('Production Environment Factors', fontweight='bold')
    ax.set_xlabel('Evaluation ID')
    ax.set_ylabel('Score')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

def _plot_variance_analysis(ax, df, stats):
    """Plot variance analysis box plot."""
    box_plot = ax.boxplot([df[df['question_type'] == qt]['numeric_score'].values 
                          for qt in stats['question_stats'].index], 
                         tick_labels=[qt.replace('_', '\n') for qt in stats['question_stats'].index],
                         patch_artist=True)
    
    for patch, outlier_count in zip(box_plot['boxes'], stats['outlier_counts']):
        if outlier_count > 2:
            patch.set_facecolor('lightcoral')
            patch.set_alpha(0.7)
        else:
            patch.set_facecolor('steelblue')
            patch.set_alpha(0.7)
    
    ax.axhline(stats['mean_score'], color='red', linestyle='-', linewidth=2, alpha=0.8, label='Overall Mean')
    ax.set_title('Judge Evaluation Variance Analysis', fontweight='bold')
    ax.set_xlabel('Question Type')
    ax.set_ylabel('Score')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=9)

def _plot_production_summary(ax, stats):
    """Plot production readiness summary."""
    ax.axis('off')
    
    summary_text = f"""PRODUCTION READINESS ASSESSMENT:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        Mean Performance: {stats['mean_score']:.2f}/10
        Standard Deviation: {stats['std_score']:.2f}
        Quality Threshold: {stats['outlier_threshold']:.2f}
        Review Required: {len(stats['outliers'])} cases ({len(stats['outliers'])/(sum(stats['quality_counts']))*100:.1f}%)

        ✓ SLA EXPECTATION: {(stats['quality_counts'][0]+stats['quality_counts'][1])/sum(stats['quality_counts'])*100:.1f}% satisfactory
        ✓ CAPACITY PLANNING: {len(stats['outliers'])} cases need human review
        ✓ MONITORING THRESHOLD: < {stats['outlier_threshold']:.1f} triggers alert
        ✓ IMPROVEMENT PRIORITY: {stats['detailed_stats'][0]['type'].replace('_', ' ').title()}

        STATISTICAL SIGNIFICANCE: Large sample 
        enables detection of ±{stats['std_score']:.1f} point changes
        """
    
    ax.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=11, 
            family='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    ax.set_title('Production Deployment Readiness', fontweight='bold')

def print_performance_summary(stats):
    """Print performance analysis summary."""
    total_samples = sum(stats['quality_counts'])
    satisfactory_pct = (stats['quality_counts'][0] + stats['quality_counts'][1]) / total_samples * 100
    outlier_pct = len(stats['outliers']) / total_samples * 100

    print("Large-scale analysis complete! Model shows realistic performance distribution")
    print(f"Mean degradation from perfect (10.0) to production reality ({stats['mean_score']:.2f})")
    print(f"{satisfactory_pct:.1f}% of cases meet quality thresholds - ready for production deployment")
    print(f"{len(stats['outliers'])} cases ({outlier_pct:.1f}%) require human review - typical for scale")
    print(f"Statistical significance achieved: can detect ±{stats['std_score']:.1f} point performance changes")

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
        print(f"   Response: {json.dumps(result['response'], indent=2)}")

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
        print(f"\nEvaluation Summary:")
        print(f"Average Score: {df_evaluations['numeric_score'].mean():.2f}")
        print(f"Score Range: {df_evaluations['numeric_score'].min():.1f} - {df_evaluations['numeric_score'].max():.1f}")

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
        if not metrics_df.empty and 'numeric_score' in metrics_df.columns:
            print(f"\n" + "=" * 80)
            print("CITIES EVALUATION ANALYSIS")
            print("=" * 80)

            print(f"\nSCORE STATISTICS:")
            print(f"  Average Score: {metrics_df['numeric_score'].mean():.2f}/10")
            print(f"  Median Score: {metrics_df['numeric_score'].median():.2f}/10")
            print(f"  Score Range: {metrics_df['numeric_score'].min():.1f} - {metrics_df['numeric_score'].max():.1f}")

            if 'question_type' in metrics_df.columns:
                print(f"\nQUESTION TYPE PERFORMANCE:")
                type_stats = metrics_df.groupby('question_type')['numeric_score'].agg(['mean', 'count'])
                for question_type, stats in type_stats.iterrows():
                    print(f"  {question_type.replace('_', ' ').title()}: {stats['mean']:.2f} avg ({int(stats['count'])} questions)")

            if 'complexity' in metrics_df.columns:
                print(f"\nCOMPLEXITY PERFORMANCE:")
                complexity_stats = metrics_df.groupby('complexity')['numeric_score'].agg(['mean', 'count'])
                for complexity, stats in complexity_stats.iterrows():
                    print(f"  {complexity}: {stats['mean']:.2f} avg ({int(stats['count'])} questions)")

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
    print(f"Overall Performance: {stats['mean_score']:.2f} ± {stats['std_score']:.2f}")
    print(f"Outliers (< {stats['outlier_threshold']:.2f}): {len(stats['outliers'])} cases")
    print("Outlier breakdown by question type:")
    print(stats['outliers'].groupby('question_type').size().sort_values(ascending=False))

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

    # Judge evaluation summary
    if df_evaluations is not None and not df_evaluations.empty:
        summary['judge'] = {
            'mean_score': df_evaluations['numeric_score'].mean(),
            'median_score': df_evaluations['numeric_score'].median(),
            'min_score': df_evaluations['numeric_score'].min(),
            'max_score': df_evaluations['numeric_score'].max(),
            'std_dev': df_evaluations['numeric_score'].std(),
            'total_evaluated': len(df_evaluations)
        }

        # Question type breakdown
        if 'question_type' in df_evaluations.columns:
            type_performance = df_evaluations.groupby('question_type')['numeric_score'].agg(['mean', 'count'])
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

        # Score visualization
        mean_score = judge['mean_score']
        score_bar_length = 20
        score_filled = int(score_bar_length * mean_score / 10)
        score_bar = "🟩" * score_filled + "⬜" * (score_bar_length - score_filled)

        print(f"Average Score: {score_bar} {mean_score:.2f}/10")
        print(f"Score Range: {judge['min_score']:.1f} - {judge['max_score']:.1f}")
        print(f"Median: {judge['median_score']:.1f} | Std Dev: {judge['std_dev']:.2f}")
        print(f"Total Evaluated: {judge['total_evaluated']}")

    # Question Type Performance
    if 'by_type' in summary:
        print("\nPERFORMANCE BY QUESTION TYPE")
        print("-" * 40)
        for q_type, stats in summary['by_type'].items():
            # Mini bar for each type
            mini_bar_length = 10
            mini_filled = int(mini_bar_length * stats['mean'] / 10)
            mini_bar = "▰" * mini_filled + "▱" * (mini_bar_length - mini_filled)

            print(f"{q_type.replace('_', ' ').title():25} {mini_bar} {stats['mean']:.1f}/10 ({int(stats['count'])} samples)")

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
    
    if plot_type == "score_distribution":
        _plot_score_distribution(plt.gca(), df, stats)
    elif plot_type == "performance_by_type":
        _plot_performance_by_type(plt.gca(), stats)
    elif plot_type == "quality_distribution":
        _plot_quality_distribution(plt.gca(), stats)
    elif plot_type == "outlier_analysis":
        _plot_outlier_analysis(plt.gca(), df, stats)
    elif plot_type == "variance_analysis":
        _plot_variance_analysis(plt.gca(), df, stats)
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