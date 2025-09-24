# Enhanced Multimodal RAGAS Implementation - Complete Tutorial

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

A comprehensive implementation of Enhanced Multimodal RAGAS (Retrieval-Augmented Generation Assessment) that combines traditional RAGAS metrics with custom multimodal evaluation using ImageBind embeddings. This system evaluates retrieval performance across text, vision, and audio modalities using the Cinepile movie dataset.

## **IMPORTANT: Download Dataset First**

**Before starting, you MUST download the required dataset:**

```bash
# Download the Cinepile dataset (487MB)
wget "https://d22xjg1p9prwde.cloudfront.net/cinepile-dataset.tar.gz"

# Extract the dataset
tar -xzf cinepile-dataset.tar.gz

# Move to the expected location
mv cinepile-dataset /home/sagemaker-user/MMRAG/
```

**Alternative download methods:**
- **Direct browser download**: [https://d22xjg1p9prwde.cloudfront.net/cinepile-dataset.tar.gz](https://d22xjg1p9prwde.cloudfront.net/cinepile-dataset.tar.gz)
- **Using curl**: `curl -O "https://d22xjg1p9prwde.cloudfront.net/cinepile-dataset.tar.gz"`

The dataset contains:
- **yt_audios/**: Audio files from movie clips
- **yt_videos_frames/**: Extracted video frames  
- **yt_text/**: Text descriptions and dialogue
- **Size**: 584MB uncompressed, 487MB compressed

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset Structure](#dataset-structure)
- [Evaluation Metrics](#evaluation-metrics)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Results Interpretation](#results-interpretation)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Citation](#citation)

## Overview

This project implements a state-of-the-art multimodal retrieval evaluation system that:

- **Combines Traditional RAGAS with Multimodal Metrics**: Integrates standard RAG evaluation with custom cross-modal assessment
- **Supports Multiple Modalities**: Evaluates text, vision, and audio retrieval simultaneously
- **Uses ImageBind Embeddings**: Leverages Meta's ImageBind for unified multimodal representations
- **Provides Comprehensive Analysis**: Offers detailed performance metrics and visualizations
- **Integrates with AWS SageMaker**: Uses cloud-based LLMs for evaluation

### Learning Objectives

By completing this tutorial, you will understand:

1. **Multimodal Embeddings**: How to create unified embeddings across text, vision, and audio using ImageBind
2. **RAGAS Integration**: How to combine traditional RAGAS metrics with custom multimodal evaluation
3. **Retrieval Strategies**: How to evaluate different retrieval approaches (text-only, vision-only, audio-only, multimodal)
4. **Performance Analysis**: How to interpret and visualize multimodal retrieval performance
5. **Strategic Insights**: When to use different retrieval strategies based on your use case

## Features

### Core Capabilities
- **Multimodal Retrieval**: Text, vision, and audio search capabilities
- **Comprehensive Metrics**: Traditional RAGAS + custom multimodal metrics
- **Efficient Search**: FAISS-based similarity search for scalability
- **Strategy Comparison**: Evaluate different retrieval approaches
- **Rich Visualizations**: Detailed performance analysis and charts
- **Modular Design**: Easy to extend and customize

### Evaluation Strategies
1. **Text-only retrieval**: Using only text embeddings
2. **Vision-only retrieval**: Using only image embeddings  
3. **Audio-only retrieval**: Using only audio embeddings
4. **Full multimodal retrieval**: Combining all three modalities

### Display Features (Fixed!)
- **Image Display**: Properly shows retrieved video frames
- **Audio Playback**: Functional audio players for retrieved clips
- **Interactive Demos**: Working multimodal retrieval demonstrations

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Cinepile      │    │    ImageBind     │    │   FAISS         │
│   Dataset       │───▶│    Model         │───▶│   Indices       │
│                 │    │                  │    │                 │
│ • Text          │    │ • Text Encoder   │    │ • Text Index    │
│ • Images        │    │ • Vision Encoder │    │ • Vision Index  │
│ • Audio         │    │ • Audio Encoder  │    │ • Audio Index   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Retrieval     │    │   Evaluation     │    │   Results       │
│   Engine        │───▶│   System         │───▶│   Analysis      │
│                 │    │                  │    │                 │
│ • Query         │    │ • RAGAS Metrics  │    │ • Performance   │
│ • Search        │    │ • Custom Metrics │    │ • Visualizations│
│ • Ranking       │    │ • LLM Integration│    │ • Comparisons   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Key Components

- **ImageBind Model**: Creates unified embeddings across text, vision, and audio modalities
- **FAISS Indices**: Enables efficient similarity search for each modality
- **RAGAS Metrics**: Provides standard RAG evaluation (answer relevancy, faithfulness, context precision/recall)
- **Custom Multimodal Metrics**: Adds modality-specific evaluation capabilities
- **SageMaker Integration**: Uses AWS SageMaker endpoints for LLM-based evaluation
- **MultimodalShowcase**: Interactive demonstration system with working media display

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- AWS Account with SageMaker access
- 8GB+ RAM
- 10GB+ disk space

### Step 1: Clone ImageBind

```bash
git clone https://github.com/facebookresearch/ImageBind.git
cd ImageBind
```

### Step 2: Install Dependencies

```bash
# Install required packages (skip ImageBind's requirements.txt due to PyTorch version conflicts)
pip install pandas numpy torch faiss-cpu scikit-learn boto3 langchain-aws langchain-core ragas datasets tqdm matplotlib seaborn ipython soundfile decord ftfy regex Pillow pytorchvideo==0.1.5 librosa
```

### Step 3: Download Dataset

```bash
# Download the Cinepile dataset
wget "https://d22xjg1p9prwde.cloudfront.net/cinepile-dataset.tar.gz"
tar -xzf cinepile-dataset.tar.gz
```

### Step 4: Configure AWS

```bash
# Configure AWS credentials for SageMaker access
aws configure
```

### Step 5: Set Environment Variables

```python
# In your notebook or script
import os
os.environ['CINEPILE_DATA_PATH'] = '/path/to/cinepile-dataset'
os.environ['SAGEMAKER_ENDPOINT_NAME'] = 'your-sagemaker-endpoint'
```

## Quick Start

### Option 1: Jupyter Notebook (Recommended)

1. **Open the main tutorial**:
   ```bash
   jupyter notebook Enhanced_Multimodal_RAGAS_Tutorial.ipynb
   ```

2. **Run the test cell first** to verify display functions work:
   - Look for the test cell after "Showcase system ready!"
   - You should see a bird image and hear bird audio

3. **Follow the step-by-step tutorial**:
   - Each cell is documented with explanations
   - Run cells in order for best results

### Option 2: Python Script

```python
# Import the utilities
from utils import load_cinepile_data, MultimodalShowcase

# Load the dataset
data_entries = load_cinepile_data()
print(f"Loaded {len(data_entries)} multimodal entries")

# Set up the showcase system
showcase = MultimodalShowcase()
showcase.setup_showcase()

# Run a quick demonstration
showcase.show_multimodal_retrieval_with_choices(question_idx=0, top_k=3)
```

### Option 3: Interactive Demo

```python
# Quick interactive demo
from utils import MultimodalShowcase

# Initialize and setup
showcase = MultimodalShowcase()
showcase.setup_showcase()

# Demo different retrieval strategies
demo_questions = [0, 1, 2]

for question_idx in demo_questions:
    print(f"\n{'='*50}")
    print(f"QUESTION {question_idx}")
    print(f"{'='*50}")
    
    # Text retrieval
    showcase.show_text_retrieval_with_choices(question_idx, top_k=3)
    
    # Vision retrieval (with actual images!)
    showcase.show_vision_retrieval_with_choices(question_idx, top_k=3)
    
    # Audio retrieval (with playable audio!)
    showcase.show_audio_retrieval_with_choices(question_idx, top_k=3)
    
    # Multimodal retrieval (images + audio + text!)
    showcase.show_multimodal_retrieval_with_choices(question_idx, top_k=3)
    
    # NEW: Custom multimodal retrieval with your own image/audio
    showcase.show_multimodal_retrieval_with_choices(
        question_idx=question_idx, 
        top_k=3,
        query_image_path="./cinepile-dataset/yt_videos_frames/1_Wy4EfdnMZ5g/frame_118.jpg",
        query_audio_path="./cinepile-dataset/yt_audios/1_Wy4EfdnMZ5g.wav"
    )
```

## Dataset Structure

The Cinepile dataset contains multimodal movie data organized as follows:

```
cinepile-dataset/
├── yt_audios/              # Audio files from movie clips
│   ├── 0_5s8dYeDZPAE.wav  # Format: {id}_{video_id}.wav
│   ├── 1_abc123def.wav
│   └── ...
├── yt_videos_frames/       # Extracted video frames
│   ├── 0_5s8dYeDZPAE/     # One directory per video
│   │   ├── frame1.jpg
│   │   ├── frame2.jpg
│   │   └── ...
│   └── ...
├── yt_text/               # Text descriptions and dialogue
│   ├── 0_5s8dYeDZPAE.txt # Corresponding text content
│   ├── 1_abc123def.txt
│   └── ...
└── README.txt             # Dataset documentation
```

### Data Loading

The `load_cinepile_data()` function automatically:
- Matches audio, text, and frame files by ID
- Selects representative frames from each video
- Creates unified data entries for multimodal processing
- Handles missing files gracefully

## Evaluation Metrics

### Traditional RAGAS Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **Answer Relevancy** | How relevant is the generated answer to the question? | 0.0 - 1.0 |
| **Faithfulness** | Is the answer faithful to the retrieved context? | 0.0 - 1.0 |
| **Context Precision** | How precise is the retrieved context? | 0.0 - 1.0 |
| **Context Recall** | How much relevant context was retrieved? | 0.0 - 1.0 |

### Custom Multimodal Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **Multimodal Faithfulness** | Cross-modal consistency of retrieved content | 0.0 - 1.0 |
| **Multimodal Relevancy** | Semantic relevance across modalities | 0.0 - 1.0 |
| **Cross-Modal Coherence** | How well do different modalities align? | 0.0 - 1.0 |
| **Modality Coverage** | How many modalities contribute to the result? | 0.0 - 1.0 |

### Retrieval Performance Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **Precision@K** | Precision at top K results | 0.0 - 1.0 |
| **Recall@K** | Recall at top K results | 0.0 - 1.0 |
| **MRR@K** | Mean Reciprocal Rank | 0.0 - 1.0 |
| **NDCG@K** | Normalized Discounted Cumulative Gain | 0.0 - 1.0 |

## Usage Examples

### Custom Multimodal Retrieval

```python
from utils import MultimodalShowcase

# Initialize showcase
showcase = MultimodalShowcase()
showcase.setup_showcase()

# Standard multimodal retrieval
showcase.show_multimodal_retrieval_with_choices(question_idx=0, top_k=3)

# Custom multimodal retrieval with your own image and audio
showcase.show_multimodal_retrieval_with_choices(
    question_idx=0, 
    top_k=3,
    query_image_path="./cinepile-dataset/yt_videos_frames/1_Wy4EfdnMZ5g/frame_118.jpg",
    query_audio_path="./cinepile-dataset/yt_audios/1_Wy4EfdnMZ5g.wav"
)
```

### Custom Evaluation

```python
from utils import MultimodalRAGASEvaluator

# Initialize evaluator
evaluator = MultimodalRAGASEvaluator(
    sagemaker_endpoint="your-endpoint",
    embeddings=embeddings
)

# Run evaluation
results = evaluator.evaluate_strategy(
    strategy_name="multimodal",
    questions=questions,
    contexts=contexts,
    answers=answers
)

print(f"Average RAGAS Score: {results['ragas_score']:.3f}")
print(f"Multimodal Faithfulness: {results['multimodal_faithfulness']:.3f}")
```

### Visualization

```python
import matplotlib.pyplot as plt
from utils import plot_evaluation_results

# Plot comparison across strategies
strategies = ['text_only', 'vision_only', 'audio_only', 'multimodal']
results = {strategy: evaluate_strategy(strategy) for strategy in strategies}

plot_evaluation_results(results)
plt.show()
```

## API Reference

### Core Functions

#### `load_cinepile_data()`
Loads and organizes the Cinepile dataset for multimodal processing.

**Returns:**
- `List[Dict]`: List of data entries with text, image, and audio paths

#### `create_embeddings(data_entries, device='cuda')`
Creates ImageBind embeddings for all modalities.

**Parameters:**
- `data_entries`: List of data entries from `load_cinepile_data()`
- `device`: Computing device ('cuda' or 'cpu')

**Returns:**
- `Dict`: Dictionary containing embeddings for each modality

#### `build_indices(embeddings)`
Builds FAISS indices for efficient similarity search.

**Parameters:**
- `embeddings`: Dictionary of embeddings from `create_embeddings()`

**Returns:**
- `Tuple`: (text_index, vision_index, audio_index, multimodal_indices)

### MultimodalShowcase Class

#### `MultimodalShowcase()`
Interactive demonstration system with working media display.

**Key Methods:**
- `setup_showcase()`: Initialize the showcase system
- `show_text_retrieval_with_choices(question_idx, top_k=3)`: Demo text retrieval
- `show_vision_retrieval_with_choices(question_idx, top_k=3, query_image_path=None)`: Demo vision retrieval with images
- `show_audio_retrieval_with_choices(question_idx, top_k=3, query_audio_path=None)`: Demo audio retrieval with playback
- `show_multimodal_retrieval_with_choices(question_idx, top_k=3, query_image_path=None, query_audio_path=None)`: Demo multimodal retrieval with optional custom inputs

### Evaluation Classes

#### `MultimodalRAGASEvaluator`
Comprehensive evaluation system combining RAGAS with multimodal metrics.

**Methods:**
- `evaluate_strategy(strategy_name, questions, contexts, answers)`: Evaluate a retrieval strategy
- `compute_multimodal_metrics(retrieved_contexts, ground_truth)`: Compute custom metrics
- `generate_report(results)`: Generate detailed evaluation report

## Results Interpretation

### Understanding the Metrics

**High Performance Indicators:**
- **RAGAS Scores > 0.7**: Good traditional RAG performance
- **Multimodal Faithfulness > 0.8**: Strong cross-modal consistency
- **Precision@5 > 0.6**: Good retrieval accuracy
- **NDCG@5 > 0.7**: Good ranking quality

**Strategy Comparison:**
- **Text-only**: Best for factual, descriptive queries
- **Vision-only**: Best for visual scene understanding
- **Audio-only**: Best for audio-specific content (music, dialogue)
- **Multimodal**: Best overall performance, especially for complex queries

### Performance Analysis

The system typically shows:
1. **Multimodal retrieval** outperforms single-modality approaches
2. **Vision-only** performs well for scene-based queries
3. **Text-only** excels at factual information retrieval
4. **Audio-only** is specialized but effective for audio content

## Troubleshooting

### Common Issues

#### 1. Images Not Displaying
**Problem**: "This XML file does not appear to have any style information"

**Solutions:**
- Clear browser cache (Ctrl+F5)
- Try incognito/private mode
- Run the test cell to verify display functions
- Check that sample files exist in `ImageBind/.assets/`

#### 2. Audio Not Playing
**Problem**: Audio players not functional

**Solutions:**
- Install librosa: `pip install librosa`
- Check browser audio permissions
- Try different browsers (Chrome, Firefox)
- Verify audio files exist and are readable

#### 3. CUDA Out of Memory
**Problem**: GPU memory errors during embedding creation

**Solutions:**
```python
# Reduce batch size
BATCH_SIZE = 8  # Instead of 32

# Use CPU if necessary
device = 'cpu'

# Clear GPU cache
import torch
torch.cuda.empty_cache()
```

#### 4. SageMaker Endpoint Issues
**Problem**: Cannot connect to SageMaker endpoint

**Solutions:**
- Verify AWS credentials: `aws sts get-caller-identity`
- Check endpoint name and region
- Ensure endpoint is deployed and in-service
- Verify IAM permissions for SageMaker access

#### 5. Dataset Loading Errors
**Problem**: Cannot find dataset files

**Solutions:**
```bash
# Verify dataset download
ls -la cinepile-dataset/

# Check expected structure
find cinepile-dataset/ -type f | head -10

# Re-download if necessary
wget "https://d22xjg1p9prwde.cloudfront.net/cinepile-dataset.tar.gz"
```

### Performance Optimization

#### For Large Datasets
```python
# Use smaller embedding dimensions
EMBEDDING_DIM = 512  # Instead of 1024

# Implement batch processing
def process_in_batches(data, batch_size=32):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

# Use approximate search
index.nprobe = 10  # Faster but less accurate
```

#### For Limited Memory
```python
# Use CPU-only mode
device = 'cpu'

# Reduce precision
embeddings = embeddings.astype(np.float16)

# Use memory mapping for large indices
index = faiss.read_index("large_index.faiss", faiss.IO_FLAG_MMAP)
```

### Development Setup

```bash
# Clone the repository
git clone https://github.com/your-repo/enhanced-multimodal-ragas.git
cd enhanced-multimodal-ragas

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 src/
black src/
```

### Areas for Contribution

- **New Modalities**: Add support for video, 3D, or other modalities
- **Metrics**: Implement additional evaluation metrics
- **Performance**: Optimize embedding creation and search
- **Documentation**: Improve tutorials and examples
- **Bug Fixes**: Fix issues and improve stability


## References

- [ImageBind Paper](https://arxiv.org/abs/2305.05665) - The foundational research
- [RAGAS Documentation](https://docs.ragas.io/) - Traditional RAG evaluation
- [FAISS Documentation](https://faiss.ai/) - Efficient similarity search
- [Multimodal AI Research](https://paperswithcode.com/task/multimodal-learning) - Latest developments
- [PyTorch Documentation](https://pytorch.org/docs/) - Deep learning framework
- [AWS SageMaker](https://docs.aws.amazon.com/sagemaker/) - Cloud ML platform
