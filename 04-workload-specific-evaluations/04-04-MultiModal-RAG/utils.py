# Global configuration
_CINEPILE_DATA_PATH = None

def set_cinepile_data_path(path):
    """Set the global Cinepile dataset path for all utilities"""
    global _CINEPILE_DATA_PATH
    _CINEPILE_DATA_PATH = path
    print(f"✅ Cinepile dataset path set to: {path}")

def get_cinepile_data_path():
    """Get the current Cinepile dataset path"""
    if _CINEPILE_DATA_PATH is None:
        raise ValueError("Cinepile dataset path not set. Please call set_cinepile_data_path() first.")
    return _CINEPILE_DATA_PATH


import boto3
from langchain_aws.chat_models.sagemaker_endpoint import ChatSagemakerEndpoint, ChatModelContentHandler
from langchain_core.messages import HumanMessage, AIMessageChunk, SystemMessage
from botocore.response import StreamingBody
import numpy as np
import pandas as pd
import os
import re
import json
import torch
import faiss
from pathlib import Path
from sklearn.preprocessing import normalize
from sklearn.metrics import ndcg_score

# ImageBind imports
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

# RAGAS imports
import ragas
from ragas.run_config import RunConfig
from ragas.metrics.base import MetricWithLLM, MetricWithEmbeddings
from ragas import evaluate
from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas.metrics import answer_relevancy, faithfulness, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

# Langchain base classes
from langchain_core.embeddings import Embeddings
from typing import List, Dict, Any, Optional, Union, Tuple
from sklearn.metrics.pairwise import cosine_similarity

class QueryStructure:
    """Class to clearly define and document query structures"""
    
    def __init__(self, strategy_name: str, modalities: List[str], 
                 text_query: str = None, image_path: str = None, 
                 audio_path: str = None, description: str = ""):
        self.strategy_name = strategy_name
        self.modalities = modalities
        self.text_query = text_query
        self.image_path = image_path
        self.audio_path = audio_path
        self.description = description
        
        # Validate that required data is provided for requested modalities
        self._validate_modalities()
    
    def _validate_modalities(self):
        """Validate that required data is available for requested modalities"""
        print(f" Validating modalities: {self.modalities}")
        if 'text' in self.modalities and (not self.text_query or self.text_query is None):
            raise ValueError("Text modality requested but no text_query provided")
        if 'vision' in self.modalities and (not self.image_path or self.image_path is None):
            raise ValueError("Vision modality requested but no image_path provided")  
        if 'audio' in self.modalities and (not self.audio_path or self.audio_path is None):
            raise ValueError("Audio modality requested but no audio_path provided")
        print(" Validation passed")
    
    def __str__(self):
        components = []
        if self.text_query:
            components.append(f"TEXT: '{self.text_query[:50]}...'")
        if self.image_path:
            components.append(f"IMAGE: {Path(self.image_path).name}")
        if self.audio_path:
            components.append(f"AUDIO: {Path(self.audio_path).name}")
        
        return f"{self.strategy_name} -> {' + '.join(components)}"
    
    def get_summary(self):
        return {
            'strategy': self.strategy_name,
            'modalities': self.modalities,
            'text_query': self.text_query[:100] + "..." if self.text_query and len(self.text_query) > 100 else self.text_query,
            'image_file': Path(self.image_path).name if self.image_path else None,
            'audio_file': Path(self.audio_path).name if self.audio_path else None,
            'description': self.description
        }

def load_cinepile_questions(dataset_path=None):
    """Load cinepile questions and metadata"""
    print("\n Loading cinepile questions...")

    if dataset_path is None:
        dataset_path = get_cinepile_data_path()
    
    cinepile_data_path = dataset_path
    loaded_selected_videos = pd.read_pickle(os.path.join(cinepile_data_path, 'selected_videos.pkl'))
    
    questions = loaded_selected_videos["question"]
    answer_keys = loaded_selected_videos["answer_key"]
    answer_key_position = loaded_selected_videos["answer_key_position"]
    choices = loaded_selected_videos["choices"]
    
    print(f" Loaded {len(questions)} questions")
    
    return {
        'questions': questions,
        'answer_keys': answer_keys,
        'answer_key_positions': answer_key_position,
        'choices': choices,
        'dataframe': loaded_selected_videos
    }

def load_cinepile_data(dataset_path=None):
    """Load cinepile dataset and organize for retrieval evaluation"""
    print("\n Loading cinepile dataset...")

    if dataset_path is None:
        dataset_path = get_cinepile_data_path()
    
    dataset_path = Path(dataset_path)
    
    # Get directories
    audio_dir = dataset_path / "yt_audios"
    text_dir = dataset_path / "yt_text"
    frames_dir = dataset_path / "yt_videos_frames"
    
    # Data containers
    data_entries = []
    
    # Process each audio file
    for audio_file in sorted(audio_dir.glob("*.wav")):
        audio_stem = audio_file.stem  # e.g., "0_5s8dYeDZPAE"
        video_id = audio_stem.split('_')[0]
        
        # Find corresponding text file
        text_file = text_dir / f"{audio_stem}.txt"
        if not text_file.exists():
            continue
            
        # Find corresponding frame directory
        frame_dir = frames_dir / audio_stem
        if not frame_dir.exists():
            continue
            
        # Get frame files
        frame_files = list(frame_dir.glob("frame*.jpg"))
        if not frame_files:
            continue
            
        # Sort frames by number
        def extract_frame_number(frame_path):
            match = re.search(r'frame(\d+)', frame_path.name)
            return int(match.group(1)) if match else 0
        
        frame_files.sort(key=extract_frame_number)
        
        # Read text content
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                text_content = f.read().strip()
        except Exception as e:
            continue
        
        # Sample 5 frames per video for comprehensive coverage
        frames_per_video = 5
        if len(frame_files) >= frames_per_video:
            sampled_frames = frame_files[::max(1, len(frame_files) // frames_per_video)]
            sampled_frames = sampled_frames[:frames_per_video]
        else:
            sampled_frames = frame_files
        
        # Create entries for each frame
        for frame_idx, frame_path in enumerate(sampled_frames):
            data_entries.append({
                'id': f"{video_id}_{frame_idx}",  # Unique identifier
                'video_id': int(video_id),  # Convert to int for matching with questions
                'text': text_content,
                'image_path': str(frame_path),
                'audio_path': str(audio_file),
                'frame_index': frame_idx,
                'total_frames': len(sampled_frames)
            })
    
    print(f" Loaded {len(data_entries)} entries from {len(set(e['video_id'] for e in data_entries))} videos")
    
    return data_entries

def create_imagebind_embeddings(model, data_entries, batch_size=8):
    """Create separate ImageBind embeddings for each modality"""
    print(f"\n Creating ImageBind embeddings for all modalities...")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Device detected is ",device)
    
    # Prepare data
    texts = [entry['text'] for entry in data_entries]
    image_paths = [entry['image_path'] for entry in data_entries]
    audio_paths = [entry['audio_path'] for entry in data_entries]

    
    # Storage for embeddings
    text_embeddings = []
    vision_embeddings = []
    audio_embeddings = []
    
    # Process in batches
    num_batches = (len(data_entries) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(data_entries))
        
        print(f"  Processing batch {batch_idx + 1}/{num_batches} (items {start_idx}-{end_idx-1})")
        
        # Prepare batch
        batch_texts = texts[start_idx:end_idx]
        batch_images = image_paths[start_idx:end_idx]
        batch_audios = audio_paths[start_idx:end_idx]
        
        # Create ImageBind inputs with error handling
        try:
            inputs = {
                ModalityType.TEXT: data.load_and_transform_text(batch_texts, device),
                ModalityType.VISION: data.load_and_transform_vision_data(batch_images, device),
                ModalityType.AUDIO: data.load_and_transform_audio_data(batch_audios, device),
            }
            
            # Check for None values before passing to model
            none_modalities = []
            for modality_name, tensor in inputs.items():
                if tensor is None:
                    none_modalities.append(modality_name)
            
            if none_modalities:
                print(f"    ⚠️  Warning: None values found for {none_modalities} in batch {batch_idx + 1}")
                print(f"    Skipping batch {batch_idx + 1}")
                
                # Create zero embeddings for this batch to maintain consistency
                batch_size_actual = len(batch_texts)
                zero_embedding = np.zeros((batch_size_actual, 1024))  # ImageBind uses 1024-dim embeddings
                text_embeddings.append(zero_embedding)
                vision_embeddings.append(zero_embedding)
                audio_embeddings.append(zero_embedding)
                continue
            
        except Exception as e:
            print(f"    ❌ Error loading data for batch {batch_idx + 1}: {e}")
            print(f"    Skipping batch {batch_idx + 1}")
            
            # Create zero embeddings for this batch
            batch_size_actual = len(batch_texts)
            zero_embedding = np.zeros((batch_size_actual, 1024))
            text_embeddings.append(zero_embedding)
            vision_embeddings.append(zero_embedding)
            audio_embeddings.append(zero_embedding)
            continue
        
        # Get embeddings
        try:
            with torch.no_grad():
                embeddings = model(inputs)
            
            # Store embeddings
            text_embeddings.append(embeddings[ModalityType.TEXT].detach().cpu().numpy())
            vision_embeddings.append(embeddings[ModalityType.VISION].detach().cpu().numpy())
            audio_embeddings.append(embeddings[ModalityType.AUDIO].detach().cpu().numpy())
            
        except Exception as e:
            print(f"    ❌ Error getting embeddings for batch {batch_idx + 1}: {e}")
            print(f"    Skipping batch {batch_idx + 1}")
            
            # Create zero embeddings for this batch
            batch_size_actual = len(batch_texts)
            zero_embedding = np.zeros((batch_size_actual, 1024))
            text_embeddings.append(zero_embedding)
            vision_embeddings.append(zero_embedding)
            audio_embeddings.append(zero_embedding)
            continue
        
        # Clear GPU memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Concatenate all batches
    if text_embeddings:
        text_embeddings = np.concatenate(text_embeddings, axis=0)
        vision_embeddings = np.concatenate(vision_embeddings, axis=0)
        audio_embeddings = np.concatenate(audio_embeddings, axis=0)
        
        print(f" Created embeddings:")
        print(f"  Text embeddings: {text_embeddings.shape}")
        print(f"  Vision embeddings: {vision_embeddings.shape}")
        print(f"  Audio embeddings: {audio_embeddings.shape}")
        
        return {
            'text': text_embeddings,
            'vision': vision_embeddings,
            'audio': audio_embeddings
        }
    else:
        raise RuntimeError("No embeddings were created successfully!")

def create_separate_indices(embeddings):
    """Create separate FAISS indices for each modality"""
    print(f"\n Creating separate FAISS indices...")
    
    indices = {}
    normalized_embeddings = {}
    
    for modality, emb_matrix in embeddings.items():
        # Normalize embeddings
        emb_norm = normalize(emb_matrix, axis=1).astype('float32')
        normalized_embeddings[modality] = emb_norm
        
        # Create FAISS index
        d = emb_norm.shape[1]
        index = faiss.IndexFlatIP(d)  # Inner product for cosine similarity
        index.add(emb_norm)
        
        indices[modality] = index
        
        print(f"  {modality.capitalize()} index: {index.ntotal} vectors (dim={d})")
    
    return indices, normalized_embeddings

def create_multimodal_indices(normalized_embeddings):
    """Create multimodal indices for different combinations"""
    print(f"\n Creating multimodal indices...")
    
    multimodal_indices = {}
    multimodal_embeddings = {}
    
    # Text + Vision combination
    print("  Creating Text + Vision index...")
    text_vision_embeddings = []
    for i in range(len(normalized_embeddings['text'])):
        combined = (
            0.5 * normalized_embeddings['text'][i] +
            0.5 * normalized_embeddings['vision'][i]
        )
        combined = normalize(combined.reshape(1, -1), axis=1)[0]
        text_vision_embeddings.append(combined)
    
    text_vision_embeddings = np.stack(text_vision_embeddings).astype('float32')
    d = text_vision_embeddings.shape[1]
    text_vision_index = faiss.IndexFlatIP(d)
    text_vision_index.add(text_vision_embeddings)
    multimodal_indices['text_vision'] = text_vision_index
    multimodal_embeddings['text_vision'] = text_vision_embeddings
    print(f"    Text+Vision index: {text_vision_index.ntotal} vectors (dim={d})")
    
    # Text + Audio combination
    print("  Creating Text + Audio index...")
    text_audio_embeddings = []
    for i in range(len(normalized_embeddings['text'])):
        combined = (
            0.5 * normalized_embeddings['text'][i] +
            0.5 * normalized_embeddings['audio'][i]
        )
        combined = normalize(combined.reshape(1, -1), axis=1)[0]
        text_audio_embeddings.append(combined)
    
    text_audio_embeddings = np.stack(text_audio_embeddings).astype('float32')
    text_audio_index = faiss.IndexFlatIP(d)
    text_audio_index.add(text_audio_embeddings)
    multimodal_indices['text_audio'] = text_audio_index
    multimodal_embeddings['text_audio'] = text_audio_embeddings
    print(f"    Text+Audio index: {text_audio_index.ntotal} vectors (dim={d})")
    
    # Full multimodal (Text + Vision + Audio)
    print("  Creating Full Multimodal index...")
    full_multimodal_embeddings = []
    for i in range(len(normalized_embeddings['text'])):
        combined = (
            1/3 * normalized_embeddings['text'][i] +
            1/3 * normalized_embeddings['vision'][i] +
            1/3 * normalized_embeddings['audio'][i]
        )
        combined = normalize(combined.reshape(1, -1), axis=1)[0]
        full_multimodal_embeddings.append(combined)
    
    full_multimodal_embeddings = np.stack(full_multimodal_embeddings).astype('float32')
    full_multimodal_index = faiss.IndexFlatIP(d)
    full_multimodal_index.add(full_multimodal_embeddings)
    multimodal_indices['full_multimodal'] = full_multimodal_index
    multimodal_embeddings['full_multimodal'] = full_multimodal_embeddings
    print(f"    Full Multimodal index: {full_multimodal_index.ntotal} vectors (dim={d})")
    
    return multimodal_indices, multimodal_embeddings

def create_query_embedding(query_structure: QueryStructure, normalized_embeddings):
    """
    Create query embedding based ONLY on the specific modality combination requested.
    This prevents processing unwanted modalities that cause system unresponsiveness.
    """
    
    print(f" Processing query with modalities: {query_structure.modalities}")
    
    # QueryStructure constructor already validates required data
    
    # Process ONLY the requested modalities
    if query_structure.modalities == ['text']:
        print("    Processing TEXT ONLY")
        # Text only - no other modalities processed
        inputs = {ModalityType.TEXT: data.load_and_transform_text([query_structure.text_query], device)}
        with torch.no_grad():
            embeddings = model(inputs)
        query_emb = embeddings[ModalityType.TEXT][0].cpu().numpy()
        
    elif query_structure.modalities == ['vision']:
        print("    Processing VISION ONLY")
        # Vision only - no other modalities processed
        inputs = {ModalityType.VISION: data.load_and_transform_vision_data([query_structure.image_path], device)}
        with torch.no_grad():
            embeddings = model(inputs)
        query_emb = embeddings[ModalityType.VISION][0].cpu().numpy()
        
    elif query_structure.modalities == ['audio']:
        print("    Processing AUDIO ONLY")
        # Audio only - no other modalities processed
        inputs = {ModalityType.AUDIO: data.load_and_transform_audio_data([query_structure.audio_path], device)}
        with torch.no_grad():
            embeddings = model(inputs)
        query_emb = embeddings[ModalityType.AUDIO][0].cpu().numpy()
        
    elif set(query_structure.modalities) == {'text', 'vision'}:
        print("    Processing TEXT + VISION ONLY")
        # Text + Vision only - no audio processed
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text([query_structure.text_query], device),
            ModalityType.VISION: data.load_and_transform_vision_data([query_structure.image_path], device)
        }
        with torch.no_grad():
            embeddings = model(inputs)
        
        text_emb = normalize(embeddings[ModalityType.TEXT][0].cpu().numpy().reshape(1, -1), axis=1)[0]
        vision_emb = normalize(embeddings[ModalityType.VISION][0].cpu().numpy().reshape(1, -1), axis=1)[0]
        
        query_emb = 0.5 * text_emb + 0.5 * vision_emb
        
    elif set(query_structure.modalities) == {'text', 'audio'}:
        print("    Processing TEXT + AUDIO ONLY")
        # Text + Audio only - no vision processed
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text([query_structure.text_query], device),
            ModalityType.AUDIO: data.load_and_transform_audio_data([query_structure.audio_path], device)
        }
        with torch.no_grad():
            embeddings = model(inputs)
        
        text_emb = normalize(embeddings[ModalityType.TEXT][0].cpu().numpy().reshape(1, -1), axis=1)[0]
        audio_emb = normalize(embeddings[ModalityType.AUDIO][0].cpu().numpy().reshape(1, -1), axis=1)[0]
        
        query_emb = 0.5 * text_emb + 0.5 * audio_emb
        
    elif set(query_structure.modalities) == {'vision', 'audio'}:
        print("    Processing VISION + AUDIO ONLY")
        # Vision + Audio only - no text processed
        inputs = {
            ModalityType.VISION: data.load_and_transform_vision_data([query_structure.image_path], device),
            ModalityType.AUDIO: data.load_and_transform_audio_data([query_structure.audio_path], device)
        }
        with torch.no_grad():
            embeddings = model(inputs)
        
        vision_emb = normalize(embeddings[ModalityType.VISION][0].cpu().numpy().reshape(1, -1), axis=1)[0]
        audio_emb = normalize(embeddings[ModalityType.AUDIO][0].cpu().numpy().reshape(1, -1), axis=1)[0]
        
        query_emb = 0.5 * vision_emb + 0.5 * audio_emb
        
    elif set(query_structure.modalities) == {'text', 'vision', 'audio'}:
        print("    Processing ALL MODALITIES")
        # Full multimodal - all modalities processed
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text([query_structure.text_query], device),
            ModalityType.VISION: data.load_and_transform_vision_data([query_structure.image_path], device),
            ModalityType.AUDIO: data.load_and_transform_audio_data([query_structure.audio_path], device)
        }
        with torch.no_grad():
            embeddings = model(inputs)
        
        text_emb = normalize(embeddings[ModalityType.TEXT][0].cpu().numpy().reshape(1, -1), axis=1)[0]
        vision_emb = normalize(embeddings[ModalityType.VISION][0].cpu().numpy().reshape(1, -1), axis=1)[0]
        audio_emb = normalize(embeddings[ModalityType.AUDIO][0].cpu().numpy().reshape(1, -1), axis=1)[0]
        
        query_emb = (1/3) * text_emb + (1/3) * vision_emb + (1/3) * audio_emb
    
    else:
        raise ValueError(f"Unsupported modality combination: {query_structure.modalities}")
    
    # Normalize final embedding
    query_emb = normalize(query_emb.reshape(1, -1), axis=1)[0]
    print(f"    Query embedding created with shape: {query_emb.shape}")
    
    return query_emb.astype('float32')

def calculate_comprehensive_metrics(retrieved_video_ids: List[int], expected_video_id: int, 
                                  distances: List[float], k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
    """Calculate comprehensive retrieval metrics"""
    
    metrics = {}
    
    # Basic position metrics
    try:
        rank = retrieved_video_ids.index(expected_video_id) + 1
        found = True
    except ValueError:
        rank = len(retrieved_video_ids) + 1
        found = False
    
    metrics['rank'] = rank
    metrics['found'] = found
    
    # Calculate metrics for different k values
    for k in k_values:
        k_retrieved = retrieved_video_ids[:k]
        
        # Precision@k (for single relevant item, this is 1/k if found, 0 otherwise)
        precision_k = 1.0 / k if expected_video_id in k_retrieved else 0.0
        
        # Recall@k (for single relevant item, this is 1 if found, 0 otherwise)
        recall_k = 1.0 if expected_video_id in k_retrieved else 0.0
        
        # MRR@k
        if expected_video_id in k_retrieved and rank <= k:
            mrr_k = 1.0 / rank
        else:
            mrr_k = 0.0
        
        # NDCG@k calculation
        # Create relevance scores (1 for correct video, 0 for others)
        relevance_scores = [1.0 if vid == expected_video_id else 0.0 for vid in k_retrieved]
        
        if sum(relevance_scores) > 0:
            # Pad or truncate to exactly k items
            if len(relevance_scores) < k:
                relevance_scores.extend([0.0] * (k - len(relevance_scores)))
            
            # Calculate NDCG
            try:
                ndcg_k = ndcg_score([relevance_scores], [relevance_scores], k=k)
            except:
                ndcg_k = 0.0
        else:
            ndcg_k = 0.0
        
        metrics[f'precision@{k}'] = precision_k
        metrics[f'recall@{k}'] = recall_k
        metrics[f'mrr@{k}'] = mrr_k
        metrics[f'ndcg@{k}'] = ndcg_k
    
    return metrics

def retrieve_with_modality(query_structure: QueryStructure, k=10) -> Tuple[List[float], List[int]]:
    """Generic retrieval function using QueryStructure with proper modality combinations"""
    
    print(f" Retrieving with modalities: {query_structure.modalities}")
    
    # Create query embedding using ONLY the requested modalities
    query_emb = create_query_embedding(query_structure, normalized_embeddings)
    
    # Determine which index to use based on modalities
    if query_structure.modalities == ['text']:
        print("    Using TEXT index")
        D, I = text_index.search(query_emb.reshape(1, -1), k)
    elif query_structure.modalities == ['vision']:
        print("    Using VISION index")
        D, I = vision_index.search(query_emb.reshape(1, -1), k)
    elif query_structure.modalities == ['audio']:
        print("    Using AUDIO index")
        D, I = audio_index.search(query_emb.reshape(1, -1), k)
    elif set(query_structure.modalities) == {'text', 'vision'}:
        print("    Using TEXT+VISION index")
        D, I = multimodal_indices['text_vision'].search(query_emb.reshape(1, -1), k)
    elif set(query_structure.modalities) == {'text', 'audio'}:
        print("    Using TEXT+AUDIO index")
        D, I = multimodal_indices['text_audio'].search(query_emb.reshape(1, -1), k)
    elif set(query_structure.modalities) == {'vision', 'audio'}:
        print("    Using VISION+AUDIO index")
        D, I = multimodal_indices['vision_audio'].search(query_emb.reshape(1, -1), k)
    elif set(query_structure.modalities) == {'text', 'vision', 'audio'}:
        print("    Using FULL MULTIMODAL index")
        D, I = multimodal_indices['full_multimodal'].search(query_emb.reshape(1, -1), k)
    else:
        raise ValueError(f"Unsupported modality combination: {query_structure.modalities}")
    
    print(f"    Retrieved {len(I[0])} results")
    return D[0].tolist(), I[0].tolist()

def prepare_ragas_data(question_results: List[Dict]) -> Dataset:
    """Prepare data in RAGAS format"""
    print("\n Preparing data for RAGAS evaluation...")
    
    ragas_data = []
    
    for result in question_results:
        question = result['question']
        expected_video = result['expected_video']
        answer_key = result['answer_key']
        choices = result['choices']
        
        for strategy_name, strategy_data in result['strategies'].items():
            retrieved_videos = strategy_data['retrieved_videos']
            
            # Get retrieved contexts (text content from retrieved videos)
            retrieved_contexts = []
            for video_id in retrieved_videos[:3]:  # Top 3 contexts
                video_entries = [e for e in data_entries if e['video_id'] == video_id]
                if video_entries:
                    retrieved_contexts.append(video_entries[0]['text'])
                else:
                    retrieved_contexts.append("No content available")
            
            # Create RAGAS entry
            ragas_entry = {
                'question': question,
                'user_input': question,  # RAGAS expects this field
                'response': answer_key,  # The expected answer
                'reference': answer_key,  # Ground truth reference
                'retrieved_contexts': retrieved_contexts,
                'strategy': strategy_name,
                'expected_video': expected_video,
                'retrieved_videos': retrieved_videos[:3]
            }
            
            ragas_data.append(ragas_entry)
    
    # Convert to RAGAS Dataset format
    dataset = Dataset.from_list(ragas_data)
    print(f" Prepared {len(ragas_data)} entries for RAGAS evaluation")
    
    return dataset

# Global variables for indices (will be set during execution)
text_index = None
vision_index = None
audio_index = None
multimodal_indices = None
normalized_embeddings = None
data_entries = None

class EnhancedMultimodalImageBindEmbeddings(Embeddings):
    """
    Enhanced ImageBind embeddings wrapper that supports multiple modalities
    and can create embeddings based on content type and strategy
    """
    
    def __init__(self, model, device, data_entries):
        self.model = model
        self.device = device
        self.data_entries = data_entries  # Access to video data for multimodal content
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed documents using text modality (default for RAGAS compatibility)
        """
        try:
            inputs = {
                ModalityType.TEXT: data.load_and_transform_text(texts, self.device)
            }
            
            with torch.no_grad():
                embeddings = self.model(inputs)
                text_embeddings = embeddings[ModalityType.TEXT].cpu().numpy()
            
            return text_embeddings.tolist()
            
        except Exception as e:
            print(f"Error in embed_documents: {e}")
            return [[0.0] * 1024 for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed query using text modality (default for RAGAS compatibility)
        """
        try:
            inputs = {
                ModalityType.TEXT: data.load_and_transform_text([text], self.device)
            }
            
            with torch.no_grad():
                embeddings = self.model(inputs)
                text_embedding = embeddings[ModalityType.TEXT][0].cpu().numpy()
            
            return text_embedding.tolist()
            
        except Exception as e:
            print(f"Error in embed_query: {e}")
            return [0.0] * 1024
    
    def embed_multimodal_query(self, query_structure: QueryStructure) -> np.ndarray:
        """
        Create embeddings based on the query structure's modalities
        """
        try:
            inputs = {}
            
            # Add modalities based on query structure
            if 'text' in query_structure.modalities and query_structure.text_query:
                inputs[ModalityType.TEXT] = data.load_and_transform_text([query_structure.text_query], self.device)
            
            if 'vision' in query_structure.modalities and query_structure.image_path:
                inputs[ModalityType.VISION] = data.load_and_transform_vision_data([query_structure.image_path], self.device)
            
            if 'audio' in query_structure.modalities and query_structure.audio_path:
                inputs[ModalityType.AUDIO] = data.load_and_transform_audio_data([query_structure.audio_path], self.device)
            
            # Get embeddings
            with torch.no_grad():
                embeddings = self.model(inputs)
            
            # Combine embeddings based on modalities
            if len(query_structure.modalities) == 1:
                # Single modality
                if 'text' in query_structure.modalities:
                    return embeddings[ModalityType.TEXT][0].cpu().numpy()
                elif 'vision' in query_structure.modalities:
                    return embeddings[ModalityType.VISION][0].cpu().numpy()
                elif 'audio' in query_structure.modalities:
                    return embeddings[ModalityType.AUDIO][0].cpu().numpy()
            else:
                # Multimodal - combine with equal weights
                combined_embedding = None
                weight = 1.0 / len(query_structure.modalities)
                
                for modality in query_structure.modalities:
                    if modality == 'text' and ModalityType.TEXT in embeddings:
                        emb = embeddings[ModalityType.TEXT][0].cpu().numpy()
                    elif modality == 'vision' and ModalityType.VISION in embeddings:
                        emb = embeddings[ModalityType.VISION][0].cpu().numpy()
                    elif modality == 'audio' and ModalityType.AUDIO in embeddings:
                        emb = embeddings[ModalityType.AUDIO][0].cpu().numpy()
                    else:
                        continue
                    
                    if combined_embedding is None:
                        combined_embedding = weight * emb
                    else:
                        combined_embedding += weight * emb
                
                return combined_embedding
                
        except Exception as e:
            print(f"Error in embed_multimodal_query: {e}")
            return np.zeros(1024)
    
    def embed_retrieved_contexts_multimodal(self, retrieved_video_ids: List[int], 
                                           query_structure: QueryStructure) -> List[np.ndarray]:
        """
        Create embeddings for retrieved contexts based on query modalities
        """
        context_embeddings = []
        
        for video_id in retrieved_video_ids:
            try:
                # Get video data
                video_entries = [e for e in self.data_entries if e['video_id'] == video_id]
                if not video_entries:
                    context_embeddings.append(np.zeros(1024))
                    continue
                
                video_entry = video_entries[0]
                inputs = {}
                
                # Add same modalities as query
                if 'text' in query_structure.modalities:
                    inputs[ModalityType.TEXT] = data.load_and_transform_text([video_entry['text']], self.device)
                
                if 'vision' in query_structure.modalities:
                    inputs[ModalityType.VISION] = data.load_and_transform_vision_data([video_entry['image_path']], self.device)
                
                if 'audio' in query_structure.modalities:
                    inputs[ModalityType.AUDIO] = data.load_and_transform_audio_data([video_entry['audio_path']], self.device)
                
                # Get embeddings
                with torch.no_grad():
                    embeddings = self.model(inputs)
                
                # Combine embeddings
                if len(query_structure.modalities) == 1:
                    if 'text' in query_structure.modalities:
                        context_embeddings.append(embeddings[ModalityType.TEXT][0].cpu().numpy())
                    elif 'vision' in query_structure.modalities:
                        context_embeddings.append(embeddings[ModalityType.VISION][0].cpu().numpy())
                    elif 'audio' in query_structure.modalities:
                        context_embeddings.append(embeddings[ModalityType.AUDIO][0].cpu().numpy())
                else:
                    # Multimodal combination
                    combined_embedding = None
                    weight = 1.0 / len(query_structure.modalities)
                    
                    for modality in query_structure.modalities:
                        if modality == 'text' and ModalityType.TEXT in embeddings:
                            emb = embeddings[ModalityType.TEXT][0].cpu().numpy()
                        elif modality == 'vision' and ModalityType.VISION in embeddings:
                            emb = embeddings[ModalityType.VISION][0].cpu().numpy()
                        elif modality == 'audio' and ModalityType.AUDIO in embeddings:
                            emb = embeddings[ModalityType.AUDIO][0].cpu().numpy()
                        else:
                            continue
                        
                        if combined_embedding is None:
                            combined_embedding = weight * emb
                        else:
                            combined_embedding += weight * emb
                    
                    context_embeddings.append(combined_embedding)
                    
            except Exception as e:
                print(f"Error embedding context for video {video_id}: {e}")
                context_embeddings.append(np.zeros(1024))
        
        return context_embeddings

class CustomMultimodalMetrics:
    """
    Custom metrics that work with multimodal embeddings
    """
    
    def __init__(self, embeddings_wrapper: EnhancedMultimodalImageBindEmbeddings):
        self.embeddings = embeddings_wrapper
    
    def calculate_multimodal_similarity(self, query_structure: QueryStructure, 
                                      retrieved_video_ids: List[int], 
                                      expected_video_id: int) -> Dict[str, float]:
        """
        Calculate similarity metrics using appropriate modalities
        """
        try:
            # Get query embedding
            query_embedding = self.embeddings.embed_multimodal_query(query_structure)
            
            # Get context embeddings
            context_embeddings = self.embeddings.embed_retrieved_contexts_multimodal(
                retrieved_video_ids[:5], query_structure
            )
            
            # Calculate similarities
            similarities = []
            for ctx_emb in context_embeddings:
                sim = cosine_similarity(
                    query_embedding.reshape(1, -1), 
                    ctx_emb.reshape(1, -1)
                )[0][0]
                similarities.append(sim)
            
            # Find expected video similarity
            expected_similarity = 0.0
            if expected_video_id in retrieved_video_ids[:5]:
                expected_idx = retrieved_video_ids[:5].index(expected_video_id)
                expected_similarity = similarities[expected_idx]
            
            return {
                'avg_similarity': np.mean(similarities),
                'max_similarity': np.max(similarities),
                'expected_similarity': expected_similarity,
                'similarities': similarities
            }
            
        except Exception as e:
            print(f"Error calculating multimodal similarity: {e}")
            return {
                'avg_similarity': 0.0,
                'max_similarity': 0.0,
                'expected_similarity': 0.0,
                'similarities': [0.0] * 5
            }
    
    def calculate_multimodal_faithfulness(self, query_structure: QueryStructure,
                                        retrieved_video_ids: List[int],
                                        expected_video_id: int,
                                        answer_key: str) -> float:
        """
        Custom faithfulness metric using multimodal embeddings
        """
        try:
            # Perfect faithfulness if correct video retrieved
            if retrieved_video_ids[0] == expected_video_id:
                return 1.0
            
            # Otherwise, calculate based on multimodal similarity
            similarity_metrics = self.calculate_multimodal_similarity(
                query_structure, retrieved_video_ids, expected_video_id
            )
            
            # Use expected similarity as faithfulness proxy
            return similarity_metrics['expected_similarity']
            
        except Exception as e:
            print(f"Error calculating multimodal faithfulness: {e}")
            return 0.0
    
    def calculate_multimodal_relevancy(self, query_structure: QueryStructure,
                                     retrieved_video_ids: List[int]) -> float:
        """
        Custom relevancy metric using multimodal embeddings
        """
        try:
            similarity_metrics = self.calculate_multimodal_similarity(
                query_structure, retrieved_video_ids, retrieved_video_ids[0]
            )
            
            # Use average similarity as relevancy
            return similarity_metrics['avg_similarity']
            
        except Exception as e:
            print(f"Error calculating multimodal relevancy: {e}")
            return 0.0



class ContentHandler(ChatModelContentHandler):
    """Content handler for SageMaker endpoint"""
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt, model_kwargs: Dict) -> bytes:
        body = {
            "messages": prompt,
            "stream": True,
            **model_kwargs
        }
        return json.dumps(body).encode("utf-8")

    def transform_output(self, output: StreamingBody) -> AIMessageChunk:
        stop_token = "[DONE]"
        try:
            all_content = []
            for line in output.iter_lines():
                if line:
                    line = line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    try:
                        json_data = json.loads(line[6:])
                    except json.JSONDecodeError as e:
                        continue
                    if json_data.get("choices", [{}])[0].get("delta", {}).get("content") == stop_token:
                        break
                    content = json_data["choices"][0]["delta"]["content"]
                    all_content.append(content)
            full_response = "".join(all_content)
            return AIMessageChunk(content=full_response)
        except Exception as e:
            return AIMessageChunk(content=f"Error processing response: {str(e)}")

def setup_enhanced_multimodal_ragas():
    """Setup enhanced multimodal RAGAS with ImageBind"""
    print(f"\n Setting up Enhanced Multimodal RAGAS...")
    print(f"   SageMaker Endpoint: {SAGEMAKER_ENDPOINT_NAME}")
    print(f"   Embeddings: Enhanced Multimodal ImageBind")
    
    # Setup SageMaker LLM
    sm = boto3.Session().client('sagemaker-runtime')
    chat_content_handler = ContentHandler()
    
    llm = ChatSagemakerEndpoint(
        name="Qwen2.5-Enhanced-RAGAS",
        endpoint_name=SAGEMAKER_ENDPOINT_NAME,
        client=sm,
        model_kwargs={
            "temperature": 0.7,
            "max_new_tokens": 1200,
            "top_p": 0.95,
            "do_sample": True
        },
        content_handler=chat_content_handler
    )
    
    # Setup Enhanced ImageBind embeddings
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    imagebind_model_instance = imagebind_model.imagebind_huge(pretrained=True).eval().to(device)
    
    embeddings = EnhancedMultimodalImageBindEmbeddings(
        model=imagebind_model_instance,
        device=device,
        data_entries=data_entries  # Pass data entries for multimodal access
    )
    
    # Setup custom multimodal metrics
    custom_metrics = CustomMultimodalMetrics(embeddings)
    
    print(" Enhanced multimodal models setup complete")
    return llm, embeddings, custom_metrics

def init_ragas_metrics_enhanced(metrics, llm, embedding):
    """Initialize RAGAS metrics with enhanced embeddings"""
    print("\n Initializing RAGAS metrics with enhanced multimodal embeddings...")
    
    for metric in metrics:
        if isinstance(metric, MetricWithLLM):
            print(f"  {metric.name} <- LLM")
            metric.llm = llm
        if isinstance(metric, MetricWithEmbeddings):
            print(f"  {metric.name} <- Enhanced Multimodal ImageBind Embeddings")
            metric.embeddings = embedding
        run_config = RunConfig()
        metric.init(run_config)
    
    print(" RAGAS metrics initialized with enhanced embeddings")

async def score_with_enhanced_ragas(query, chunks, answer, metrics, query_structure, 
                                   retrieved_video_ids, expected_video_id, custom_metrics):
    """
    Score using both RAGAS (for text) and custom multimodal metrics
    """
    scores = {}
    
    # 1. Traditional RAGAS metrics (text-based)
    print("     Calculating traditional RAGAS metrics...")
    for metric in metrics:
        sample = SingleTurnSample(
            user_input=query,
            retrieved_contexts=chunks,
            response=answer,
            reference=chunks[0] if chunks else ""
        )
        try:
            scores[f"ragas_{metric.name}"] = await metric.single_turn_ascore(sample)
            print(f"     ragas_{metric.name}: {scores[f'ragas_{metric.name}']:.3f}")
        except Exception as e:
            print(f"     Error calculating ragas_{metric.name}: {e}")
            scores[f"ragas_{metric.name}"] = 0.0
    
    # 2. Custom multimodal metrics
    print("     Calculating custom multimodal metrics...")
    try:
        # Multimodal similarity metrics
        similarity_metrics = custom_metrics.calculate_multimodal_similarity(
            query_structure, retrieved_video_ids, expected_video_id
        )
        scores.update({f"multimodal_{k}": v for k, v in similarity_metrics.items() if k != 'similarities'})
        
        # Multimodal faithfulness
        multimodal_faithfulness = custom_metrics.calculate_multimodal_faithfulness(
            query_structure, retrieved_video_ids, expected_video_id, answer
        )
        scores['multimodal_faithfulness'] = multimodal_faithfulness
        
        # Multimodal relevancy
        multimodal_relevancy = custom_metrics.calculate_multimodal_relevancy(
            query_structure, retrieved_video_ids
        )
        scores['multimodal_relevancy'] = multimodal_relevancy
        
        print(f"     multimodal_faithfulness: {multimodal_faithfulness:.3f}")
        print(f"     multimodal_relevancy: {multimodal_relevancy:.3f}")
        print(f"     multimodal_avg_similarity: {similarity_metrics['avg_similarity']:.3f}")
        
    except Exception as e:
        print(f"     Error calculating multimodal metrics: {e}")
        scores.update({
            'multimodal_faithfulness': 0.0,
            'multimodal_relevancy': 0.0,
            'multimodal_avg_similarity': 0.0,
            'multimodal_max_similarity': 0.0,
            'multimodal_expected_similarity': 0.0
        })
    
    return scores

def run_enhanced_multimodal_evaluation():
    """Run the complete enhanced multimodal evaluation"""
    print(f"\n RUNNING ENHANCED MULTIMODAL EVALUATION")
    print("=" * 60)
    
    # Setup enhanced models
    try:
        llm, embeddings, custom_metrics = setup_enhanced_multimodal_ragas()
        
        # Define RAGAS metrics (text-based)
        ragas_metrics = [
            answer_relevancy,
            faithfulness,
            context_precision,
            context_recall,
        ]
        
        # Initialize RAGAS metrics
        init_ragas_metrics_enhanced(
            ragas_metrics,
            llm=LangchainLLMWrapper(llm),
            embedding=LangchainEmbeddingsWrapper(embeddings),
        )
        
    except Exception as e:
        print(f" Failed to setup enhanced models: {e}")
        import traceback
        traceback.print_exc()
        return []
    
    # Load questions
    question_data = load_cinepile_questions()
    questions = question_data['questions']
    choices = question_data['choices']
    answer_keys = question_data['answer_keys']
    dataframe = question_data['dataframe']
    
    # Results storage
    detailed_results = []
    
    # Test questions
    num_questions = min(NUM_QUESTIONS_TO_TEST, len(questions))
    print(f" Testing {num_questions} questions with enhanced multimodal evaluation...")
    
    for idx in range(num_questions):
        question = questions.iloc[idx]
        choice_list = choices.iloc[idx]
        answer_key = answer_keys.iloc[idx]
        expected_video_id = idx
        movie_name = dataframe.iloc[idx]['movie_name']
        
        print(f"\n Question {idx+1}/{num_questions}: {movie_name}")
        print(f"   Q: {question}")
        
        # Get reference video data
        video_entries = [entry for entry in data_entries if entry['video_id'] == expected_video_id]
        if not video_entries:
            continue
        
        reference_entry = video_entries[0]
        question_with_choices = f"{question} Answer choices: {', '.join(choice_list)}"
        
        # Test different strategies with appropriate modalities
        query_strategies = [
            QueryStructure(
                strategy_name="text_only",
                modalities=["text"],
                text_query=question_with_choices,
                description="Uses question + choices text for retrieval"
            ),
            QueryStructure(
                strategy_name="vision_only",
                modalities=["vision"],
                image_path=reference_entry['image_path'],
                description="Uses only visual content for retrieval"
            ),
            QueryStructure(
                strategy_name="audio_only",
                modalities=["audio"],
                audio_path=reference_entry['audio_path'],
                description="Uses only audio content for retrieval"
            ),
            QueryStructure(
                strategy_name="multimodal_full",
                modalities=["text", "vision", "audio"],
                text_query=question_with_choices,
                image_path=reference_entry['image_path'],
                audio_path=reference_entry['audio_path'],
                description="Uses all modalities"
            )
        ]
        
        question_results = {
            'question_idx': idx,
            'movie_name': movie_name,
            'question': question,
            'expected_video': expected_video_id,
            'answer_key': answer_key,
            'choices': choice_list,
            'strategies': {}
        }
        
        for query_structure in query_strategies:
            try:
                print(f"\n    Testing {query_structure.strategy_name} with {query_structure.modalities}...")
                
                # Perform retrieval
                distances, indices = retrieve_with_modality(query_structure, k=10)
                retrieved_video_ids = [data_entries[i]['video_id'] for i in indices]
                
                # Calculate standard IR metrics
                standard_metrics = calculate_comprehensive_metrics(retrieved_video_ids, expected_video_id, distances)
                
                # Get retrieved contexts for RAGAS (text-based)
                retrieved_contexts = []
                for video_id in retrieved_video_ids[:3]:
                    video_entries_for_context = [e for e in data_entries if e['video_id'] == video_id]
                    if video_entries_for_context:
                        retrieved_contexts.append(video_entries_for_context[0]['text'])
                    else:
                        retrieved_contexts.append("No content available")
                
                # Calculate enhanced metrics (RAGAS + Custom Multimodal)
                print(f"    Calculating enhanced metrics for {query_structure.strategy_name}...")
                try:
                    import asyncio
                    enhanced_scores = asyncio.run(score_with_enhanced_ragas(
                        question, retrieved_contexts, answer_key, ragas_metrics,
                        query_structure, retrieved_video_ids, expected_video_id, custom_metrics
                    ))
                except Exception as e:
                    print(f"    Enhanced scoring failed: {e}")
                    enhanced_scores = {}
                
                # Store results
                strategy_results = {
                    'query_structure': query_structure.get_summary(),
                    'retrieved_videos': retrieved_video_ids[:5],
                    'distances': distances[:5],
                    'standard_metrics': standard_metrics,
                    'enhanced_scores': enhanced_scores
                }
                
                question_results['strategies'][query_structure.strategy_name] = strategy_results
                
                # Print summary
                top1_correct = "" if retrieved_video_ids[0] == expected_video_id else ""
                print(f"    {query_structure.strategy_name}: {top1_correct} Top-1")
                print(f"       Standard: P@1={standard_metrics['precision@1']:.3f}, MRR@5={standard_metrics['mrr@5']:.3f}")
                if enhanced_scores:
                    print(f"       Enhanced: Multimodal Faithfulness={enhanced_scores.get('multimodal_faithfulness', 0):.3f}")
                
            except Exception as e:
                print(f"    {query_structure.strategy_name}: Error - {e}")
                import traceback
                traceback.print_exc()
        
        detailed_results.append(question_results)
    
    return detailed_results

def print_enhanced_summary(results):
    """Print comprehensive summary with both standard and enhanced metrics"""
    print(f"\n ENHANCED MULTIMODAL EVALUATION SUMMARY")
    print("=" * 60)
    
    if not results:
        print(" No results to summarize")
        return
    
    # Aggregate results by strategy
    strategy_metrics = {}
    
    for result in results:
        for strategy_name, strategy_data in result['strategies'].items():
            if strategy_name not in strategy_metrics:
                strategy_metrics[strategy_name] = {
                    'standard_metrics': {},
                    'enhanced_scores': {},
                    'count': 0
                }
            
            # Aggregate standard metrics
            for metric_name, value in strategy_data['standard_metrics'].items():
                if metric_name not in strategy_metrics[strategy_name]['standard_metrics']:
                    strategy_metrics[strategy_name]['standard_metrics'][metric_name] = []
                strategy_metrics[strategy_name]['standard_metrics'][metric_name].append(value)
            
            # Aggregate enhanced scores
            for metric_name, value in strategy_data['enhanced_scores'].items():
                if metric_name not in strategy_metrics[strategy_name]['enhanced_scores']:
                    strategy_metrics[strategy_name]['enhanced_scores'][metric_name] = []
                strategy_metrics[strategy_name]['enhanced_scores'][metric_name].append(value)
            
            strategy_metrics[strategy_name]['count'] += 1
    
    # Print results for each strategy
    for strategy_name, metrics in strategy_metrics.items():
        print(f"\n {strategy_name.upper().replace('_', ' ')}:")
        print(f"   Questions tested: {metrics['count']}")
        
        # Standard IR metrics
        print("    Standard IR Metrics:")
        key_standard = ['precision@1', 'recall@1', 'mrr@5', 'ndcg@5']
        for metric_name in key_standard:
            if metric_name in metrics['standard_metrics']:
                avg_value = np.mean(metrics['standard_metrics'][metric_name])
                print(f"      {metric_name}: {avg_value:.3f}")
        
        # Enhanced metrics
        print("    RAGAS Metrics (Text-based):")
        ragas_metrics = [k for k in metrics['enhanced_scores'].keys() if k.startswith('ragas_')]
        for metric_name in ragas_metrics:
            avg_value = np.mean(metrics['enhanced_scores'][metric_name])
            print(f"      {metric_name}: {avg_value:.3f}")
        
        print("    Custom Multimodal Metrics:")
        multimodal_metrics = [k for k in metrics['enhanced_scores'].keys() if k.startswith('multimodal_')]
        for metric_name in multimodal_metrics:
            avg_value = np.mean(metrics['enhanced_scores'][metric_name])
            print(f"      {metric_name}: {avg_value:.3f}")
    
    return strategy_metrics

def analyze_modality_effectiveness(results):
    """Analyze which modalities are most effective"""
    print(f"\n MODALITY EFFECTIVENESS ANALYSIS")
    print("=" * 50)
    
    modality_performance = {}
    
    for result in results:
        for strategy_name, strategy_data in result['strategies'].items():
            modalities = strategy_data['query_structure']['modalities']
            modality_key = '+'.join(sorted(modalities))
            
            if modality_key not in modality_performance:
                modality_performance[modality_key] = {
                    'precision@1': [],
                    'multimodal_faithfulness': [],
                    'multimodal_relevancy': []
                }
            
            # Standard metrics
            modality_performance[modality_key]['precision@1'].append(
                strategy_data['standard_metrics']['precision@1']
            )
            
            # Enhanced metrics
            enhanced_scores = strategy_data['enhanced_scores']
            modality_performance[modality_key]['multimodal_faithfulness'].append(
                enhanced_scores.get('multimodal_faithfulness', 0.0)
            )
            modality_performance[modality_key]['multimodal_relevancy'].append(
                enhanced_scores.get('multimodal_relevancy', 0.0)
            )
    
    # Print analysis
    for modality_combo, metrics in modality_performance.items():
        print(f"\n {modality_combo.upper()} MODALITY:")
        print(f"   Precision@1: {np.mean(metrics['precision@1']):.3f}")
        print(f"   Multimodal Faithfulness: {np.mean(metrics['multimodal_faithfulness']):.3f}")
        print(f"   Multimodal Relevancy: {np.mean(metrics['multimodal_relevancy']):.3f}")



def display_audio_info(audio_path: str, rank: str = ""):
    """Display audio file information instead of trying to play"""
    audio_file = Path(audio_path)
    
    print(f"    Audio File: {audio_file.name}")
    print(f"    Path: {audio_path}")
    
    # Check if file exists
    if audio_file.exists():
        # Get file size
        file_size = audio_file.stat().st_size
        print(f"    Size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        print(f"    File exists and ready for playback")
    else:
        print(f"    File not found!")
    
    # Provide playback instructions
    print(f"    To play: Use your system's audio player")
    print(f"    Command: vlc '{audio_path}' or mpv '{audio_path}'")

class MultimodalRetrievalSystem:
    """
    Comprehensive multimodal retrieval system that handles all aspects of 
    embedding creation, indexing, and retrieval with proper dependency management.
    """
    
    def __init__(self, model=None, device=None, data_entries=None, 
                 embeddings=None, normalized_embeddings=None, 
                 indices=None, multimodal_indices=None, multimodal_embeddings=None):
        """
        Initialize the retrieval system with core components
        
        Args:
            model: ImageBind model instance (optional, will be loaded if not provided)
            device: Device to use for computations (optional, will be auto-detected)
            data_entries: Data entries for the dataset (optional, will be loaded if not provided)
            embeddings: Pre-created embeddings dict (optional, will be created if not provided)
            normalized_embeddings: Pre-created normalized embeddings dict (optional)
            indices: Pre-created FAISS indices dict (optional, will be created if not provided)
            multimodal_indices: Pre-created multimodal indices dict (optional)
            multimodal_embeddings: Pre-created multimodal embeddings dict (optional)
        """
        self.model = model
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_entries = data_entries
        
        # Use provided embeddings and indices or initialize empty dicts
        self.embeddings = embeddings or {}
        self.normalized_embeddings = normalized_embeddings or {}
        self.indices = indices or {}
        self.multimodal_indices = multimodal_indices or {}
        self.multimodal_embeddings = multimodal_embeddings or {}
        
        # Question data
        self.question_data = None
        
        # Check what components were provided
        provided_components = []
        if model is not None:
            provided_components.append("model")
        if embeddings:
            provided_components.append("embeddings")
        if indices:
            provided_components.append("indices")
        if multimodal_indices:
            provided_components.append("multimodal_indices")
        
        if provided_components:
            print(f" MultimodalRetrievalSystem initialized with device: {self.device}")
            print(f"    Pre-loaded components: {', '.join(provided_components)}")
        else:
            print(f" MultimodalRetrievalSystem initialized with device: {self.device}")
            print(f"     No pre-loaded components - will create as needed")
    
    def load_model(self):
        """Load ImageBind model if not already loaded"""
        if self.model is None:
            print(" Loading ImageBind model...")
            from imagebind.models import imagebind_model
            self.model = imagebind_model.imagebind_huge(pretrained=True).eval().to(self.device)
            print(" ImageBind model loaded successfully")
        else:
            print(" Using pre-loaded ImageBind model")
        return self.model
    
    def load_data_entries(self):
        """Load data entries if not already loaded"""
        if self.data_entries is None:
            print(" Loading data entries...")
            self.data_entries = load_cinepile_data()
            print(f" Loaded {len(self.data_entries)} data entries")
        return self.data_entries
    
    def load_question_data(self):
        """Load question data if not already loaded"""
        if self.question_data is None:
            print(" Loading question data...")
            self.question_data = load_cinepile_questions()
            print(f" Loaded {len(self.question_data['questions'])} questions")
        return self.question_data
    
    def create_embeddings(self, batch_size=8):
        """Create ImageBind embeddings for all modalities"""
        # Check if embeddings are already provided
        if self.embeddings:
            print(" Using pre-created embeddings")
            for modality, emb_matrix in self.embeddings.items():
                print(f"  {modality.capitalize()} embeddings: {emb_matrix.shape}")
            return self.embeddings
        
        # Ensure dependencies are loaded
        self.load_model()
        self.load_data_entries()
        
        print(f"\n Creating ImageBind embeddings for all modalities...")
        
        # Prepare data
        texts = [entry['text'] for entry in self.data_entries]
        image_paths = [entry['image_path'] for entry in self.data_entries]
        audio_paths = [entry['audio_path'] for entry in self.data_entries]
        
        # Storage for embeddings
        text_embeddings = []
        vision_embeddings = []
        audio_embeddings = []
        
        # Process in batches
        num_batches = (len(self.data_entries) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(self.data_entries))
            
            print(f"  Processing batch {batch_idx + 1}/{num_batches} (items {start_idx}-{end_idx-1})")
            
            # Prepare batch
            batch_texts = texts[start_idx:end_idx]
            batch_images = image_paths[start_idx:end_idx]
            batch_audios = audio_paths[start_idx:end_idx]
            
            # Create ImageBind inputs
            inputs = {
                ModalityType.TEXT: data.load_and_transform_text(batch_texts, self.device),
                ModalityType.VISION: data.load_and_transform_vision_data(batch_images, self.device),
                ModalityType.AUDIO: data.load_and_transform_audio_data(batch_audios, self.device),
            }
            
            # Get embeddings
            with torch.no_grad():
                embeddings = self.model(inputs)
            
            # Store embeddings
            text_embeddings.append(embeddings[ModalityType.TEXT].detach().cpu().numpy())
            vision_embeddings.append(embeddings[ModalityType.VISION].detach().cpu().numpy())
            audio_embeddings.append(embeddings[ModalityType.AUDIO].detach().cpu().numpy())
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Concatenate all batches
        self.embeddings = {
            'text': np.concatenate(text_embeddings, axis=0),
            'vision': np.concatenate(vision_embeddings, axis=0),
            'audio': np.concatenate(audio_embeddings, axis=0)
        }
        
        print(f" Created embeddings:")
        for modality, emb_matrix in self.embeddings.items():
            print(f"  {modality.capitalize()} embeddings: {emb_matrix.shape}")
        
        return self.embeddings
    
    def create_indices(self):
        """Create FAISS indices for all modalities"""
        # Check if indices are already provided
        if self.indices and self.multimodal_indices:
            print(" Using pre-created indices")
            for modality, index in self.indices.items():
                print(f"  {modality.capitalize()} index: {index.ntotal} vectors")
            for modality, index in self.multimodal_indices.items():
                print(f"  {modality.capitalize()} index: {index.ntotal} vectors")
            return self.indices, self.normalized_embeddings
        
        # Ensure embeddings exist
        if not self.embeddings:
            self.create_embeddings()
        
        print(f"\n Creating FAISS indices...")
        
        # Create separate indices for each modality (only if not provided)
        if not self.indices:
            for modality, emb_matrix in self.embeddings.items():
                # Normalize embeddings
                emb_norm = normalize(emb_matrix, axis=1).astype('float32')
                self.normalized_embeddings[modality] = emb_norm
                
                # Create FAISS index
                d = emb_norm.shape[1]
                index = faiss.IndexFlatIP(d)  # Inner product for cosine similarity
                index.add(emb_norm)
                
                self.indices[modality] = index
                print(f"  {modality.capitalize()} index: {index.ntotal} vectors (dim={d})")
        else:
            print("  Using pre-created single-modality indices")
        
        # Create multimodal indices (only if not provided)
        if not self.multimodal_indices:
            self._create_multimodal_indices()
        else:
            print("  Using pre-created multimodal indices")
        
        return self.indices, self.normalized_embeddings
    
    def _create_multimodal_indices(self):
        """Create multimodal indices for different combinations"""
        print(f" Creating multimodal indices...")
        
        # Ensure normalized embeddings exist
        if not self.normalized_embeddings:
            for modality, emb_matrix in self.embeddings.items():
                self.normalized_embeddings[modality] = normalize(emb_matrix, axis=1).astype('float32')
        
        # Text + Vision combination
        print("  Creating Text + Vision index...")
        text_vision_embeddings = []
        for i in range(len(self.normalized_embeddings['text'])):
            combined = (
                0.5 * self.normalized_embeddings['text'][i] +
                0.5 * self.normalized_embeddings['vision'][i]
            )
            combined = normalize(combined.reshape(1, -1), axis=1)[0]
            text_vision_embeddings.append(combined)
        
        text_vision_embeddings = np.stack(text_vision_embeddings).astype('float32')
        d = text_vision_embeddings.shape[1]
        text_vision_index = faiss.IndexFlatIP(d)
        text_vision_index.add(text_vision_embeddings)
        self.multimodal_indices['text_vision'] = text_vision_index
        self.multimodal_embeddings['text_vision'] = text_vision_embeddings
        print(f"    Text+Vision index: {text_vision_index.ntotal} vectors (dim={d})")
        
        # Text + Audio combination
        print("  Creating Text + Audio index...")
        text_audio_embeddings = []
        for i in range(len(self.normalized_embeddings['text'])):
            combined = (
                0.5 * self.normalized_embeddings['text'][i] +
                0.5 * self.normalized_embeddings['audio'][i]
            )
            combined = normalize(combined.reshape(1, -1), axis=1)[0]
            text_audio_embeddings.append(combined)
        
        text_audio_embeddings = np.stack(text_audio_embeddings).astype('float32')
        text_audio_index = faiss.IndexFlatIP(d)
        text_audio_index.add(text_audio_embeddings)
        self.multimodal_indices['text_audio'] = text_audio_index
        self.multimodal_embeddings['text_audio'] = text_audio_embeddings
        print(f"    Text+Audio index: {text_audio_index.ntotal} vectors (dim={d})")
        
        # Full multimodal (Text + Vision + Audio)
        print("  Creating Full Multimodal index...")
        full_multimodal_embeddings = []
        for i in range(len(self.normalized_embeddings['text'])):
            combined = (
                1/3 * self.normalized_embeddings['text'][i] +
                1/3 * self.normalized_embeddings['vision'][i] +
                1/3 * self.normalized_embeddings['audio'][i]
            )
            combined = normalize(combined.reshape(1, -1), axis=1)[0]
            full_multimodal_embeddings.append(combined)
        
        full_multimodal_embeddings = np.stack(full_multimodal_embeddings).astype('float32')
        full_multimodal_index = faiss.IndexFlatIP(d)
        full_multimodal_index.add(full_multimodal_embeddings)
        self.multimodal_indices['full_multimodal'] = full_multimodal_index
        self.multimodal_embeddings['full_multimodal'] = full_multimodal_embeddings
        print(f"    Full Multimodal index: {full_multimodal_index.ntotal} vectors (dim={d})")
    
    def create_query_embedding(self, query_structure: QueryStructure):
        """Create query embedding based on the specific modality combination"""
        # Ensure model is loaded
        self.load_model()
        
        if query_structure.modalities == ['text']:
            # Text only
            inputs = {ModalityType.TEXT: data.load_and_transform_text([query_structure.text_query], self.device)}
            with torch.no_grad():
                embeddings = self.model(inputs)
            query_emb = embeddings[ModalityType.TEXT][0].cpu().numpy()
            
        elif query_structure.modalities == ['vision']:
            # Vision only
            inputs = {ModalityType.VISION: data.load_and_transform_vision_data([query_structure.image_path], self.device)}
            with torch.no_grad():
                embeddings = self.model(inputs)
            query_emb = embeddings[ModalityType.VISION][0].cpu().numpy()
            
        elif query_structure.modalities == ['audio']:
            # Audio only
            inputs = {ModalityType.AUDIO: data.load_and_transform_audio_data([query_structure.audio_path], self.device)}
            with torch.no_grad():
                embeddings = self.model(inputs)
            query_emb = embeddings[ModalityType.AUDIO][0].cpu().numpy()
            
        elif set(query_structure.modalities) == {'text', 'vision'}:
            # Text + Vision only
            inputs = {
                ModalityType.TEXT: data.load_and_transform_text([query_structure.text_query], self.device),
                ModalityType.VISION: data.load_and_transform_vision_data([query_structure.image_path], self.device)
            }
            with torch.no_grad():
                embeddings = self.model(inputs)
            
            text_emb = normalize(embeddings[ModalityType.TEXT][0].cpu().numpy().reshape(1, -1), axis=1)[0]
            vision_emb = normalize(embeddings[ModalityType.VISION][0].cpu().numpy().reshape(1, -1), axis=1)[0]
            
            query_emb = 0.5 * text_emb + 0.5 * vision_emb
            
        elif set(query_structure.modalities) == {'text', 'audio'}:
            # Text + Audio only
            inputs = {
                ModalityType.TEXT: data.load_and_transform_text([query_structure.text_query], self.device),
                ModalityType.AUDIO: data.load_and_transform_audio_data([query_structure.audio_path], self.device)
            }
            with torch.no_grad():
                embeddings = self.model(inputs)
            
            text_emb = normalize(embeddings[ModalityType.TEXT][0].cpu().numpy().reshape(1, -1), axis=1)[0]
            audio_emb = normalize(embeddings[ModalityType.AUDIO][0].cpu().numpy().reshape(1, -1), axis=1)[0]
            
            query_emb = 0.5 * text_emb + 0.5 * audio_emb
            
        elif set(query_structure.modalities) == {'text', 'vision', 'audio'}:
            # Full multimodal
            inputs = {
                ModalityType.TEXT: data.load_and_transform_text([query_structure.text_query], self.device),
                ModalityType.VISION: data.load_and_transform_vision_data([query_structure.image_path], self.device),
                ModalityType.AUDIO: data.load_and_transform_audio_data([query_structure.audio_path], self.device)
            }
            with torch.no_grad():
                embeddings = self.model(inputs)
            
            text_emb = normalize(embeddings[ModalityType.TEXT][0].cpu().numpy().reshape(1, -1), axis=1)[0]
            vision_emb = normalize(embeddings[ModalityType.VISION][0].cpu().numpy().reshape(1, -1), axis=1)[0]
            audio_emb = normalize(embeddings[ModalityType.AUDIO][0].cpu().numpy().reshape(1, -1), axis=1)[0]
            
            query_emb = (1/3) * text_emb + (1/3) * vision_emb + (1/3) * audio_emb
        
        # Normalize final embedding
        query_emb = normalize(query_emb.reshape(1, -1), axis=1)[0]
        return query_emb.astype('float32')
    
    def retrieve_with_modality(self, query_structure: QueryStructure, k=10):
        """Generic retrieval function using QueryStructure with proper modality combinations"""
        # Ensure indices are created
        if not self.indices:
            self.create_indices()
        
        # Create query embedding
        query_emb = self.create_query_embedding(query_structure)
        
        # Determine which index to use
        if query_structure.modalities == ['text']:
            D, I = self.indices['text'].search(query_emb.reshape(1, -1), k)
        elif query_structure.modalities == ['vision']:
            D, I = self.indices['vision'].search(query_emb.reshape(1, -1), k)
        elif query_structure.modalities == ['audio']:
            D, I = self.indices['audio'].search(query_emb.reshape(1, -1), k)
        elif set(query_structure.modalities) == {'text', 'vision'}:
            D, I = self.multimodal_indices['text_vision'].search(query_emb.reshape(1, -1), k)
        elif set(query_structure.modalities) == {'text', 'audio'}:
            D, I = self.multimodal_indices['text_audio'].search(query_emb.reshape(1, -1), k)
        elif set(query_structure.modalities) == {'text', 'vision', 'audio'}:
            D, I = self.multimodal_indices['full_multimodal'].search(query_emb.reshape(1, -1), k)
        else:
            raise ValueError(f"Unsupported modality combination: {query_structure.modalities}")
        
        return D[0].tolist(), I[0].tolist()
    
    def get_question_with_choices(self, question_idx: int):
        """Get question with choices formatted properly"""
        # Ensure question data is loaded
        self.load_question_data()
        
        if question_idx >= len(self.question_data['questions']):
            return "What is happening in this scene?", "Unknown", 0, []
        
        question = self.question_data['questions'].iloc[question_idx]
        choices = self.question_data['choices'].iloc[question_idx]
        answer_key = self.question_data['answer_keys'].iloc[question_idx]
        answer_position = self.question_data['answer_key_positions'].iloc[question_idx]
        
        # Format choices nicely
        choices_text = ", ".join([f"({chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
        
        question_with_choices = f"{question} Answer choices: {choices_text}"
        
        return question_with_choices, answer_key, answer_position, choices
    
    def show_text_retrieval_with_choices(self, question_idx, top_k=3):
        """Demonstrate text retrieval with choices in prompt"""
        # Ensure all dependencies are loaded
        self.load_data_entries()
        
        print(f"\n TEXT RETRIEVAL DEMONSTRATION (WITH CHOICES)")
        print("=" * 70)
        
        # Get question with choices
        question_with_choices, answer_key, answer_position, choices = self.get_question_with_choices(question_idx)
        
        print(f" Question: {self.question_data['questions'].iloc[question_idx]}")
        print(f" Choices: {choices}")
        print(f" Correct Answer: ({chr(65+answer_position)}) {answer_key}")
        print(f" Full Query: '{question_with_choices}'")
        print("=" * 70)
        
        # Create query structure
        query_structure = QueryStructure(
            strategy_name="text_demo_with_choices",
            modalities=["text"],
            text_query=question_with_choices,
            description="Text retrieval with choices"
        )

        print("Query Structure is {}".format(query_structure))
        print("Query structure modalities is {}".format(query_structure.modalities))
        
        # Perform retrieval
        distances, indices_result = self.retrieve_with_modality(query_structure, k=top_k)
        retrieved_video_ids = [self.data_entries[i]['video_id'] for i in indices_result]
        
        print(f"\n TOP {top_k} TEXT RETRIEVALS:")
        print("-" * 70)
        
        for rank, (video_id, distance) in enumerate(zip(retrieved_video_ids, distances), 1):
            # Get video entry
            video_entry = next(e for e in self.data_entries if e['video_id'] == video_id)
            
            # Check if this is the correct video
            correct_marker = " CORRECT!" if video_id == question_idx else ""
            
            print(f"\n RANK {rank} | Video {video_id} | Score: {distance:.3f} {correct_marker}")
            print(" FULL TEXT CONTENT:")
            print("" + "" * 78 + "")
            
            # Format text content nicely
            text_lines = video_entry['text'].split('\n')
            for line in text_lines:
                if len(line) > 76:
                    # Wrap long lines
                    words = line.split()
                    current_line = ""
                    for word in words:
                        if len(current_line + word) > 76:
                            print(f" {current_line:<76} ")
                            current_line = word + " "
                        else:
                            current_line += word + " "
                    if current_line.strip():
                        print(f" {current_line.strip():<76} ")
                else:
                    print(f" {line:<76} ")
            
    def show_vision_retrieval_with_context(self, question_idx: int, top_k: int = 3):
        """Demonstrate vision retrieval with question context"""
        # Ensure all dependencies are loaded
        self.load_data_entries()
        
        print(f"\n VISION RETRIEVAL DEMONSTRATION (WITH CONTEXT)")
        print("=" * 70)
        
        # Get question info
        question_with_choices, answer_key, answer_position, choices = self.get_question_with_choices(question_idx)
        
        # Get reference video data
        video_entries = [entry for entry in self.data_entries if entry['video_id'] == question_idx]
        if not video_entries:
            print(f" No video data found for question {question_idx}")
            return
        
        reference_entry = video_entries[question_idx]
        
        print(f" Question: {self.question_data['questions'].iloc[question_idx]}")
        print(f" Choices: {choices}")
        print(f" Correct Answer: ({chr(65+answer_position)}) {answer_key}")
        print(f" Query Image: {Path(reference_entry['image_path']).name}")
        print("=" * 70)
        
        # Create query structure
        query_structure = QueryStructure(
            strategy_name="vision_demo_with_context",
            modalities=["vision"],
            image_path=reference_entry['image_path'],
            description="Vision retrieval with context"
        )
        
        print("Query Structure is {}".format(query_structure))
        print("Query structure modalities is {}".format(query_structure.modalities))
        
        # Perform retrieval
        distances, indices = self.retrieve_with_modality(query_structure, k=top_k)
        retrieved_video_ids = [self.data_entries[i]['video_id'] for i in indices]
        
        print(f"\n TOP {top_k} VISION RETRIEVALS:")
        print("-" * 70)
        
        # Show query image
        print(f"\n QUERY IMAGE:")
        try:
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
            
            plt.figure(figsize=(6, 4))
            img = mpimg.imread(reference_entry['image_path'])
            plt.imshow(img)
            plt.title(f"Query: {Path(reference_entry['image_path']).name}\nQuestion: {self.question_data['questions'].iloc[question_idx][:50]}...", 
                     fontsize=12, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"    Query image: {reference_entry['image_path']}")
            print(f"    Could not display: {e}")
        
        # Show retrieved images
        print(f"\n RETRIEVED IMAGES:")
        
        try:
            # Create subplot for retrieved images
            fig, axes = plt.subplots(1, min(top_k, 3), figsize=(15, 5))
            if top_k == 1:
                axes = [axes]
            elif top_k == 2:
                axes = list(axes)
            
            for rank, (video_id, distance) in enumerate(zip(retrieved_video_ids[:3], distances[:3])):
                video_entry = next(e for e in self.data_entries if e['video_id'] == video_id)
                
                correct_marker = " CORRECT!" if video_id == question_idx else ""
                print(f"   Rank {rank+1}: Video {video_id} | Score: {distance:.3f} | {Path(video_entry['image_path']).name} {correct_marker}")
                
                try:
                    img = mpimg.imread(video_entry['image_path'])
                    axes[rank].imshow(img)
                    title = f"Rank {rank+1} - Video {video_id}\nScore: {distance:.3f}"
                    if video_id == question_idx:
                        title += "\n CORRECT!"
                    axes[rank].set_title(title, fontsize=10, fontweight='bold')
                    axes[rank].axis('off')
                except Exception as e:
                    axes[rank].text(0.5, 0.5, f"Image Error\n{e}", 
                                  ha='center', va='center', transform=axes[rank].transAxes)
                    axes[rank].set_title(f"Rank {rank+1} - Error")
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("    Matplotlib not available for image display")
            # Fallback to text-only display
            for rank, (video_id, distance) in enumerate(zip(retrieved_video_ids, distances), 1):
                video_entry = next(e for e in self.data_entries if e['video_id'] == video_id)
                correct_marker = " CORRECT!" if video_id == question_idx else ""
                
                print(f"\n RANK {rank} | Video {video_id} | Score: {distance:.3f} {correct_marker}")
                print(f" Image Path: {video_entry['image_path']}")
                print(f" Text Content Preview: {video_entry['text'][:100]}...")
        except Exception as e:
            print(f"    Could not create image display: {e}")
            # Fallback to text-only display
            for rank, (video_id, distance) in enumerate(zip(retrieved_video_ids, distances), 1):
                video_entry = next(e for e in self.data_entries if e['video_id'] == video_id)
                correct_marker = " CORRECT!" if video_id == question_idx else ""
                
                print(f"\n RANK {rank} | Video {video_id} | Score: {distance:.3f} {correct_marker}")
                print(f" Image Path: {video_entry['image_path']}")
                print(f" Text Content Preview: {video_entry['text'][:100]}...")
    
    def show_audio_retrieval_with_context(self, question_idx: int, top_k: int = 3):
        """Demonstrate audio retrieval with question context"""
        # Ensure all dependencies are loaded
        self.load_data_entries()
        
        print(f"\n AUDIO RETRIEVAL DEMONSTRATION (WITH CONTEXT)")
        print("=" * 70)
        
        # Get question info
        question_with_choices, answer_key, answer_position, choices = self.get_question_with_choices(question_idx)
        
        # Get reference video data
        video_entries = [entry for entry in self.data_entries if entry['video_id'] == question_idx]
        if not video_entries:
            print(f" No video data found for question {question_idx}")
            return
        
        reference_entry = video_entries[0]
        
        print(f" Question: {self.question_data['questions'].iloc[question_idx]}")
        print(f" Choices: {choices}")
        print(f" Correct Answer: ({chr(65+answer_position)}) {answer_key}")
        print(f" Query Audio: {Path(reference_entry['audio_path']).name}")
        print("=" * 70)
        
        # Create query structure
        query_structure = QueryStructure(
            strategy_name="audio_demo_with_context",
            modalities=["audio"],
            audio_path=reference_entry['audio_path'],
            description="Audio retrieval with context"
        )
        
        print("Query Structure is {}".format(query_structure))
        print("Query structure modalities is {}".format(query_structure.modalities))
        
        # Perform retrieval
        distances, indices = self.retrieve_with_modality(query_structure, k=top_k)
        retrieved_video_ids = [self.data_entries[i]['video_id'] for i in indices]
        
        print(f"\n TOP {top_k} AUDIO RETRIEVALS:")
        print("-" * 70)
        
        # Display audio information
        print(f"\n QUERY AUDIO:")
        self._display_audio_info(reference_entry['audio_path'], "Query")
        
        print(f"\n RETRIEVED AUDIO FILES:")
        for rank, (video_id, distance) in enumerate(zip(retrieved_video_ids, distances), 1):
            video_entry = next(e for e in self.data_entries if e['video_id'] == video_id)
            correct_marker = " CORRECT!" if video_id == question_idx else ""
            
            print(f"\n RANK {rank} | Video {video_id} | Score: {distance:.3f} {correct_marker}")
            self._display_audio_info(video_entry['audio_path'], f"Rank {rank}")
            print(f" Text Content Preview: {video_entry['text'][:100]}...")
    
    def _display_audio_info(self, audio_path: str, label: str = ""):
        """Display audio file information"""
        audio_file = Path(audio_path)
        
        print(f"    {label} Audio File: {audio_file.name}")
        print(f"    Path: {audio_path}")
        
        # Check if file exists
        if audio_file.exists():
            # Get file size
            file_size = audio_file.stat().st_size
            print(f"    Size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
            print(f"    File exists and ready for playback")
        else:
            print(f"    File not found!")
        
    def setup_showcase(self):
        """Setup the showcase by ensuring all components are ready"""
        print(" Setting up multimodal showcase...")
        
        # Ensure all components are loaded
        self.load_model()
        self.load_data_entries()
        self.load_question_data()
        
        # Ensure indices are created
        if not self.indices or not self.multimodal_indices:
            self.create_indices()
        
        print(" Showcase setup complete!")
        return self
    
    def show_multimodal_retrieval_with_choices(self, question_idx: int, top_k: int = 3):
        """Demonstrate multimodal retrieval with question choices and full output display"""
        # Ensure all dependencies are loaded
        self.load_data_entries()
        
        print(f"\n MULTIMODAL RETRIEVAL DEMONSTRATION (WITH CHOICES)")
        print("=" * 70)
        
        # Get question info
        question_with_choices, answer_key, answer_position, choices = self.get_question_with_choices(question_idx)
        
        # Get reference video data
        video_entries = [entry for entry in self.data_entries if entry['video_id'] == question_idx]
        if not video_entries:
            print(f" No video data found for question {question_idx}")
            return
        
        reference_entry = video_entries[0]
        
        print(f" Question: {self.question_data['questions'].iloc[question_idx]}")
        print(f" Choices: {choices}")
        print(f" Correct Answer: ({chr(65+answer_position)}) {answer_key}")
        print(f" Query: Text + Vision + Audio")
        print("=" * 70)
        
        # Create multimodal query structure
        query_structure = QueryStructure(
            strategy_name="multimodal_demo_with_choices",
            modalities=["text", "vision", "audio"],
            text_query=question_with_choices,
            image_path=reference_entry['image_path'],
            audio_path=reference_entry['audio_path'],
            description="Full multimodal retrieval with choices"
        )
        
        print("Query Structure is {}".format(query_structure))
        print("Query structure modalities is {}".format(query_structure.modalities))
        
        # Perform retrieval
        distances, indices = self.retrieve_with_modality(query_structure, k=top_k)
        retrieved_video_ids = [self.data_entries[i]['video_id'] for i in indices]
        
        print(f"\n TOP {top_k} MULTIMODAL RETRIEVALS:")
        print("-" * 70)
        
        # Show comprehensive results
        self._show_comprehensive_retrieval_results(
            retrieved_video_ids, distances, question_idx, top_k, "MULTIMODAL"
        )
    
    def _show_comprehensive_retrieval_results(self, retrieved_video_ids, distances, 
                                            question_idx, top_k, retrieval_type):
        """Show comprehensive retrieval results with all modality outputs"""
        
        for rank, (video_id, distance) in enumerate(zip(retrieved_video_ids, distances), 1):
            # Get video entry
            video_entry = next(e for e in self.data_entries if e['video_id'] == video_id)
            
            # Check if this is the correct video
            correct_marker = " CORRECT!" if video_id == question_idx else ""
            
            print(f"\n RANK {rank} | Video {video_id} | Score: {distance:.3f} {correct_marker}")
            print("=" * 60)
            
            # 1. Show text content
            self._display_text_content(video_entry['text'], rank)
            
            # 2. Show image
            self._display_image_content(video_entry['image_path'], rank, video_id)
            
            # 3. Show audio info
            self._display_audio_content(video_entry['audio_path'], rank, video_id)
            
            print("=" * 60)
    
    def _display_text_content(self, text_content, rank):
        """Display text content in a formatted way"""
        print(f"\n RANK {rank} - TEXT CONTENT:")
        print("" + "" * 78 + "")
        
        # Format text content nicely
        text_lines = text_content.split('\n')
        for line in text_lines:
            if len(line) > 76:
                # Wrap long lines
                words = line.split()
                current_line = ""
                for word in words:
                    if len(current_line + word) > 76:
                        print(f" {current_line:<76} ")
                        current_line = word + " "
                    else:
                        current_line += word + " "
                if current_line.strip():
                    print(f" {current_line.strip():<76} ")
            else:
                print(f" {line:<76} ")
        print("" + "" * 78 + "")
    
    def _display_image_content(self, image_path, rank, video_id):
        """Display image content with SageMaker notebook compatibility"""
        print(f"\n RANK {rank} - IMAGE CONTENT:")
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
            from IPython.display import display
            
            # Create figure for this specific image
            plt.figure(figsize=(8, 6))
            img = mpimg.imread(image_path)
            plt.imshow(img)
            plt.title(f"Rank {rank} - Video {video_id}\nImage: {Path(image_path).name}", 
                     fontsize=12, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            
            # Display in notebook
            plt.show()
            
        except ImportError:
            print(f"    Image Path: {image_path}")
            print(f"    Filename: {Path(image_path).name}")
            print(f"    Matplotlib not available - showing path only")
        except Exception as e:
            print(f"    Image Path: {image_path}")
            print(f"    Filename: {Path(image_path).name}")
            print(f"    Could not display image: {e}")
    
    def _display_audio_content(self, audio_path, rank, video_id):
        """Display audio content with SageMaker notebook playback capability"""
        print(f"\n RANK {rank} - AUDIO CONTENT:")
        
        audio_file = Path(audio_path)
        print(f"    Audio Path: {audio_path}")
        print(f"    Filename: {audio_file.name}")
        
        # Check if file exists and get info
        if audio_file.exists():
            file_size = audio_file.stat().st_size
            print(f"    Size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
            print(f"    File exists and ready for playback")
            
            # Try to create audio player for SageMaker notebook
            try:
                from IPython.display import Audio, display
                
                print(f"    AUDIO PLAYER:")
                # Create audio player widget
                audio_widget = Audio(audio_path, autoplay=False)
                display(audio_widget)
                
            except ImportError:
                print(f"    IPython Audio not available")
                self._show_audio_playback_instructions(audio_path)
            except Exception as e:
                print(f"    Could not create audio player: {e}")
                self._show_audio_playback_instructions(audio_path)
        else:
            print(f"    File not found!")
    
    def _show_audio_playback_instructions(self, audio_path):
        """Show instructions for audio playback"""
        print(f"    Manual Playback Options:")
        print(f"      • VLC: vlc '{audio_path}'")
        print(f"      • MPV: mpv '{audio_path}'")
        print(f"      • Python: import soundfile; soundfile.play('{audio_path}')")
    
    def run_comprehensive_demo_with_choices(self, demo_questions=None):
        """Run comprehensive demo showing all retrieval types with choices"""
        
        if demo_questions is None:
            demo_questions = [0, 1, 2]  # Default to first 3 questions
        
        print(f"\n COMPREHENSIVE MULTIMODAL DEMO")
        print("=" * 70)
        print("This demo shows all retrieval types with full output display:")
        print(" Text retrieval with formatted content")
        print(" Vision retrieval with image display")
        print(" Audio retrieval with playback capability")
        print(" Multimodal retrieval combining all modalities")
        print("=" * 70)
        
        for question_idx in demo_questions:
            # Get question info
            question = self.question_data['questions'].iloc[question_idx]
            movie_name = self.question_data['dataframe'].iloc[question_idx]['movie_name']
            
            print(f"\n" + "" * 30)
            print(f"DEMO QUESTION {question_idx}: {movie_name}")
            print(f"Question: {question}")
            print("" * 60)
            
            # 1. Text Retrieval Demo
            print(f"\n" + "" * 20 + " TEXT RETRIEVAL " + "" * 20)
            try:
                self.show_text_retrieval_with_choices(question_idx, top_k=3)
            except Exception as e:
                print(f" Text retrieval error: {e}")
            
            # 2. Vision Retrieval Demo
            print(f"\n" + "" * 20 + " VISION RETRIEVAL " + "" * 19)
            try:
                self.show_vision_retrieval_with_context(question_idx, top_k=3)
            except Exception as e:
                print(f" Vision retrieval error: {e}")
            
            # 3. Audio Retrieval Demo
            print(f"\n" + "" * 20 + " AUDIO RETRIEVAL " + "" * 20)
            try:
                self.show_audio_retrieval_with_context(question_idx, top_k=3)
            except Exception as e:
                print(f" Audio retrieval error: {e}")
            
            # 4. Multimodal Retrieval Demo
            print(f"\n" + "" * 18 + " MULTIMODAL RETRIEVAL " + "" * 17)
            try:
                self.show_multimodal_retrieval_with_choices(question_idx, top_k=3)
            except Exception as e:
                print(f" Multimodal retrieval error: {e}")
            
            print(f"\n" + "" * 70)
        
        print(f"\n COMPREHENSIVE DEMO COMPLETED!")
        print(f" All retrieval types demonstrated with full output display")
    
    def create_showcase_instance(self):
        """Create a showcase instance with all components ready"""
        print(" CREATING MULTIMODAL RAG SHOWCASE INSTANCE")
        print("=" * 70)
        print("FEATURES INCLUDED:")
        print(" Audio files show info and playback widgets")
        print(" Questions include choices in prompts")
        print(" Shows correct answer and whether retrieval found it")
        print(" Uses actual cinepile questions with context")
        print(" Displays images inline in notebooks")
        print(" Comprehensive text content formatting")
        print("=" * 70)
        
        # Setup showcase
        self.setup_showcase()
        
        print(f"\n AVAILABLE DEMO METHODS:")
        print("• show_text_retrieval_with_choices(question_idx, top_k=3)")
        print("• show_vision_retrieval_with_context(question_idx, top_k=3)")
        print("• show_audio_retrieval_with_context(question_idx, top_k=3)")
        print("• show_multimodal_retrieval_with_choices(question_idx, top_k=3)")
        print("• run_comprehensive_demo_with_choices([0, 1, 2])")
        
        return self

# Legacy function for backward compatibility - updated to work with the new system
def show_text_retrieval_with_choices(index, question_idx, top_k=3):
    """Demonstrate text retrieval with choices in prompt - Legacy wrapper"""
    # Create a retrieval system instance
    retrieval_system = MultimodalRetrievalSystem()
    
    # Use the class method
    retrieval_system.show_text_retrieval_with_choices(question_idx, top_k)

# Updated retrieve_with_modality function for backward compatibility
def retrieve_with_modality(query_structure: QueryStructure, index=None, normalized_embeddings=None, 
                          model=None, device=None, k=10) -> Tuple[List[float], List[int]]:
    """Legacy wrapper for retrieve_with_modality - now uses the class-based approach"""
    # Create a retrieval system instance
    retrieval_system = MultimodalRetrievalSystem(model=model, device=device)
    
    # Use the class method
    return retrieval_system.retrieve_with_modality(query_structure, k=k)

def main(model=None, device=None, data_entries=None, embeddings=None, 
         normalized_embeddings=None, indices=None, multimodal_indices=None):
    """Main function to run the comprehensive multimodal showcase"""
    
    print(" STARTING COMPREHENSIVE MULTIMODAL RAG SHOWCASE")
    print("=" * 70)
    print("FEATURES INCLUDED:")
    print(" Audio files show info and playback widgets in notebooks")
    print(" Questions include choices in prompts")
    print(" Shows correct answer and whether retrieval found it")
    print(" Uses actual cinepile questions with context")
    print(" Displays images inline in SageMaker notebooks")
    print(" Comprehensive text content formatting")
    print(" Memory-efficient component reuse")
    print("=" * 70)
    
    # Create showcase instance with pre-created components
    showcase = MultimodalRetrievalSystem(
        model=model,
        device=device,
        data_entries=data_entries,
        embeddings=embeddings,
        normalized_embeddings=normalized_embeddings,
        indices=indices,
        multimodal_indices=multimodal_indices
    )
    
    # Setup showcase
    showcase.setup_showcase()
    
    # Run comprehensive demo with choices
    showcase.run_comprehensive_demo_with_choices()
    
    print(f"\n INDIVIDUAL DEMO METHODS AVAILABLE:")
    print("You can also run individual demos:")
    print("• showcase.show_text_retrieval_with_choices(question_idx, top_k=3)")
    print("• showcase.show_vision_retrieval_with_context(question_idx, top_k=3)")
    print("• showcase.show_audio_retrieval_with_context(question_idx, top_k=3)")
    print("• showcase.show_multimodal_retrieval_with_choices(question_idx, top_k=3)")
    print("• showcase.run_comprehensive_demo_with_choices([0, 1, 2])")
    
    return showcase

def create_showcase(model=None, device=None, data_entries=None, embeddings=None, 
                   normalized_embeddings=None, indices=None, multimodal_indices=None):
    """Create a showcase instance - convenience function"""
    return main(model, device, data_entries, embeddings, normalized_embeddings, 
                indices, multimodal_indices)


def create_query_embedding_fixed(query_structure: QueryStructure, normalized_embeddings=None):
    """
    FIXED: Create query embedding based ONLY on the specific modality combination requested.
    This prevents processing unwanted modalities that cause system unresponsiveness.
    """
    
    print(f" Processing query with modalities: {query_structure.modalities}")
    
    # Validate that we have the required data for the requested modalities
    if 'text' in query_structure.modalities and not query_structure.text_query:
        raise ValueError("Text modality requested but no text_query provided")
    if 'vision' in query_structure.modalities and not query_structure.image_path:
        raise ValueError("Vision modality requested but no image_path provided")  
    if 'audio' in query_structure.modalities and not query_structure.audio_path:
        raise ValueError("Audio modality requested but no audio_path provided")
    
    # Process ONLY the requested modalities
    if query_structure.modalities == ['text']:
        print("    Processing TEXT ONLY")
        # Text only - no other modalities processed
        inputs = {ModalityType.TEXT: data.load_and_transform_text([query_structure.text_query], device)}
        with torch.no_grad():
            embeddings = model(inputs)
        query_emb = embeddings[ModalityType.TEXT][0].cpu().numpy()
        
    elif query_structure.modalities == ['vision']:
        print("    Processing VISION ONLY")
        # Vision only - no other modalities processed
        inputs = {ModalityType.VISION: data.load_and_transform_vision_data([query_structure.image_path], device)}
        with torch.no_grad():
            embeddings = model(inputs)
        query_emb = embeddings[ModalityType.VISION][0].cpu().numpy()
        
    elif query_structure.modalities == ['audio']:
        print("    Processing AUDIO ONLY")
        # Audio only - no other modalities processed
        inputs = {ModalityType.AUDIO: data.load_and_transform_audio_data([query_structure.audio_path], device)}
        with torch.no_grad():
            embeddings = model(inputs)
        query_emb = embeddings[ModalityType.AUDIO][0].cpu().numpy()
        
    elif set(query_structure.modalities) == {'text', 'vision'}:
        print("    Processing TEXT + VISION ONLY")
        # Text + Vision only - no audio processed
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text([query_structure.text_query], device),
            ModalityType.VISION: data.load_and_transform_vision_data([query_structure.image_path], device)
        }
        with torch.no_grad():
            embeddings = model(inputs)
        
        text_emb = normalize(embeddings[ModalityType.TEXT][0].cpu().numpy().reshape(1, -1), axis=1)[0]
        vision_emb = normalize(embeddings[ModalityType.VISION][0].cpu().numpy().reshape(1, -1), axis=1)[0]
        
        query_emb = 0.5 * text_emb + 0.5 * vision_emb
        
    elif set(query_structure.modalities) == {'text', 'audio'}:
        print("    Processing TEXT + AUDIO ONLY")
        # Text + Audio only - no vision processed
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text([query_structure.text_query], device),
            ModalityType.AUDIO: data.load_and_transform_audio_data([query_structure.audio_path], device)
        }
        with torch.no_grad():
            embeddings = model(inputs)
        
        text_emb = normalize(embeddings[ModalityType.TEXT][0].cpu().numpy().reshape(1, -1), axis=1)[0]
        audio_emb = normalize(embeddings[ModalityType.AUDIO][0].cpu().numpy().reshape(1, -1), axis=1)[0]
        
        query_emb = 0.5 * text_emb + 0.5 * audio_emb
        
    elif set(query_structure.modalities) == {'vision', 'audio'}:
        print("    Processing VISION + AUDIO ONLY")
        # Vision + Audio only - no text processed
        inputs = {
            ModalityType.VISION: data.load_and_transform_vision_data([query_structure.image_path], device),
            ModalityType.AUDIO: data.load_and_transform_audio_data([query_structure.audio_path], device)
        }
        with torch.no_grad():
            embeddings = model(inputs)
        
        vision_emb = normalize(embeddings[ModalityType.VISION][0].cpu().numpy().reshape(1, -1), axis=1)[0]
        audio_emb = normalize(embeddings[ModalityType.AUDIO][0].cpu().numpy().reshape(1, -1), axis=1)[0]
        
        query_emb = 0.5 * vision_emb + 0.5 * audio_emb
        
    elif set(query_structure.modalities) == {'text', 'vision', 'audio'}:
        print("    Processing ALL MODALITIES")
        # Full multimodal - all modalities processed
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text([query_structure.text_query], device),
            ModalityType.VISION: data.load_and_transform_vision_data([query_structure.image_path], device),
            ModalityType.AUDIO: data.load_and_transform_audio_data([query_structure.audio_path], device)
        }
        with torch.no_grad():
            embeddings = model(inputs)
        
        text_emb = normalize(embeddings[ModalityType.TEXT][0].cpu().numpy().reshape(1, -1), axis=1)[0]
        vision_emb = normalize(embeddings[ModalityType.VISION][0].cpu().numpy().reshape(1, -1), axis=1)[0]
        audio_emb = normalize(embeddings[ModalityType.AUDIO][0].cpu().numpy().reshape(1, -1), axis=1)[0]
        
        query_emb = (1/3) * text_emb + (1/3) * vision_emb + (1/3) * audio_emb
    
    else:
        raise ValueError(f"Unsupported modality combination: {query_structure.modalities}")
    
    # Normalize final embedding
    query_emb = normalize(query_emb.reshape(1, -1), axis=1)[0]
    print(f"    Query embedding created with shape: {query_emb.shape}")
    
    return query_emb.astype('float32')

def retrieve_with_modality_fixed(query_structure: QueryStructure, k=10) -> Tuple[List[float], List[int]]:
    """FIXED: Generic retrieval function using QueryStructure with proper modality combinations"""
    
    print(f" Retrieving with modalities: {query_structure.modalities}")
    
    # Create query embedding using ONLY the requested modalities
    query_emb = create_query_embedding_fixed(query_structure, normalized_embeddings)
    
    # Determine which index to use
    if query_structure.modalities == ['text']:
        print("    Using TEXT index")
        D, I = text_index.search(query_emb.reshape(1, -1), k)
    elif query_structure.modalities == ['vision']:
        print("    Using VISION index")
        D, I = vision_index.search(query_emb.reshape(1, -1), k)
    elif query_structure.modalities == ['audio']:
        print("    Using AUDIO index")
        D, I = audio_index.search(query_emb.reshape(1, -1), k)
    elif set(query_structure.modalities) == {'text', 'vision'}:
        print("    Using TEXT+VISION index")
        D, I = multimodal_indices['text_vision'].search(query_emb.reshape(1, -1), k)
    elif set(query_structure.modalities) == {'text', 'audio'}:
        print("    Using TEXT+AUDIO index")
        D, I = multimodal_indices['text_audio'].search(query_emb.reshape(1, -1), k)
    elif set(query_structure.modalities) == {'text', 'vision', 'audio'}:
        print("    Using FULL MULTIMODAL index")
        D, I = multimodal_indices['full_multimodal'].search(query_emb.reshape(1, -1), k)
    else:
        raise ValueError(f"Unsupported modality combination: {query_structure.modalities}")

    print(f"    Retrieved {len(I[0])} results")
    return D[0].tolist(), I[0].tolist()

# =============================================================================
# MULTIMODAL SHOWCASE CLASS - Added for notebook usage
# =============================================================================

class MultimodalShowcase:
    """
    Multimodal showcase class with fixed modality processing
    Uses existing embeddings and indices - no recreation needed!
    """
    
    def __init__(self, data_entries=None, question_data=None, 
                 text_index=None, vision_index=None, audio_index=None, 
                 multimodal_indices=None, normalized_embeddings=None):
        """
        Initialize with existing variables from your notebook
        
        Args:
            data_entries: Your existing data_entries
            question_data: Your existing question_data  
            text_index: Your existing text_index
            vision_index: Your existing vision_index
            audio_index: Your existing audio_index
            multimodal_indices: Your existing multimodal_indices
            normalized_embeddings: Your existing normalized_embeddings
        """
        self.data_entries = data_entries
        self.question_data = question_data
        self.text_index = text_index
        self.vision_index = vision_index
        self.audio_index = audio_index
        self.multimodal_indices = multimodal_indices
        self.normalized_embeddings = normalized_embeddings
        self.setup_complete = False
        
    def setup_showcase(self, data_entries_param=None, question_data_param=None, 
                      text_index_param=None, vision_index_param=None, audio_index_param=None, 
                      multimodal_indices_param=None, normalized_embeddings_param=None):
        """
        Setup using existing variables - no recreation!
        
        If variables not provided in constructor, pass them here.
        If not provided at all, will try to use global variables.
        """
        print(" Setting up Multimodal Showcase with existing variables...")
        
        # Use provided variables or fall back to constructor or globals
        if data_entries_param is not None:
            self.data_entries = data_entries_param
            print(" Using provided data_entries")
        elif self.data_entries is None:
            # Try to use global variable
            if 'data_entries' in globals():
                self.data_entries = globals()['data_entries']
                print(" Using existing global data_entries")
            else:
                print(" Loading cinepile data...")
                self.data_entries = load_cinepile_data()
        else:
            print(" Using constructor data_entries")
            
        if question_data_param is not None:
            self.question_data = question_data_param
            print(" Using provided question_data")
        elif self.question_data is None:
            if 'question_data' in globals():
                self.question_data = globals()['question_data']
                print(" Using existing global question_data")
            else:
                print(" Loading cinepile questions...")
                self.question_data = load_cinepile_questions()
        else:
            print(" Using constructor question_data")
            
        # Set indices (use provided or global)
        if text_index_param is not None:
            self.text_index = text_index_param
            print(" Using provided text_index")
        elif self.text_index is None:
            if 'text_index' in globals():
                self.text_index = globals()['text_index']
                print(" Using existing global text_index")
            else:
                print(" No text_index found - please provide it")
        else:
            print(" Using constructor text_index")
                
        if vision_index_param is not None:
            self.vision_index = vision_index_param
            print(" Using provided vision_index")
        elif self.vision_index is None:
            if 'vision_index' in globals():
                self.vision_index = globals()['vision_index']
                print(" Using existing global vision_index")
            else:
                print(" No vision_index found - please provide it")
        else:
            print(" Using constructor vision_index")
                
        if audio_index_param is not None:
            self.audio_index = audio_index_param
            print(" Using provided audio_index")
        elif self.audio_index is None:
            if 'audio_index' in globals():
                self.audio_index = globals()['audio_index']
                print(" Using existing global audio_index")
            else:
                print(" No audio_index found - please provide it")
        else:
            print(" Using constructor audio_index")
                
        if multimodal_indices_param is not None:
            self.multimodal_indices = multimodal_indices_param
            print(" Using provided multimodal_indices")
        elif self.multimodal_indices is None:
            if 'multimodal_indices' in globals():
                self.multimodal_indices = globals()['multimodal_indices']
                print(" Using existing global multimodal_indices")
            else:
                print(" No multimodal_indices found - please provide it")
        else:
            print(" Using constructor multimodal_indices")
                
        if normalized_embeddings_param is not None:
            self.normalized_embeddings = normalized_embeddings_param
            print(" Using provided normalized_embeddings")
        elif self.normalized_embeddings is None:
            if 'normalized_embeddings' in globals():
                self.normalized_embeddings = globals()['normalized_embeddings']
                print(" Using existing global normalized_embeddings")
            else:
                print(" No normalized_embeddings found - please provide it")
        else:
            print(" Using constructor normalized_embeddings")
        
        # Update global variables for the retrieve_with_modality function
        if self.text_index is not None:
            globals()['text_index'] = self.text_index
        if self.vision_index is not None:
            globals()['vision_index'] = self.vision_index
        if self.audio_index is not None:
            globals()['audio_index'] = self.audio_index
        if self.multimodal_indices is not None:
            globals()['multimodal_indices'] = self.multimodal_indices
        if self.normalized_embeddings is not None:
            globals()['normalized_embeddings'] = self.normalized_embeddings
        if self.data_entries is not None:
            globals()['data_entries'] = self.data_entries
        
        self.setup_complete = True
        print(" Multimodal showcase setup complete - using existing variables!")
        print(" No embeddings or indices recreated - using your existing ones!")
        
    def show_text_retrieval_with_choices(self, question_idx: int, top_k: int = 3):
        """Show text retrieval results with full text content display"""
        if not self.setup_complete:
            print(" Please run setup_showcase() first!")
            return
            
        if question_idx >= len(self.question_data['questions']):
            print(f" Question index {question_idx} out of range!")
            return
            
        # Get question text
        question_text = self.question_data['questions'].iloc[question_idx]
        
        print(f" TEXT RETRIEVAL WITH CHOICES")
        print(f"Question: {question_text}")
        print("-" * 50)
        
        # Create query structure for TEXT ONLY
        query_structure = QueryStructure(
            strategy_name=f"text_question_{question_idx}",
            modalities=["text"],  # ONLY TEXT
            text_query=question_text,
            description=f"Text retrieval for question {question_idx}"
        )
        
        print(" Processing text-only query...")
        print("   (Watch for 'Processing TEXT ONLY' message)")
        
        # Perform retrieval using FIXED functions
        distances, indices = retrieve_with_modality(query_structure, k=top_k)
        retrieved_video_ids = [self.data_entries[i]['video_id'] for i in indices]
        
        print(f"\n Top {top_k} Text Retrieval Results:")
        for rank, (video_id, distance) in enumerate(zip(retrieved_video_ids, distances), 1):
            video_entry = next(e for e in self.data_entries if e['video_id'] == video_id)
            
            print(f"\n Rank {rank} (Score: {distance:.3f})")
            print(f" Video: {video_id}")
            print(f" FULL TEXT CONTENT:")
            print("=" * 60)
            
            # Display FULL text content (not truncated)
            text_content = video_entry.get('text_content', video_entry.get('text', 'No text content available'))
            print(text_content)
            print("=" * 60)
        
        print(f" TEXT-ONLY processing complete (no vision/audio processed)")
    
    def show_vision_retrieval_with_choices(self, question_idx: int, top_k: int = 3, query_image_path: str = None):
        """Show vision retrieval results with actual image display"""
        if not self.setup_complete:
            print(" Please run setup_showcase() first!")
            return
            
        if question_idx >= len(self.question_data['questions']):
            print(f" Question index {question_idx} out of range!")
            return
            
        # Use provided query image or default to first video's image
        if query_image_path is None:
            sample_image = self.data_entries[0]['image_path']
        else:
            sample_image = query_image_path
        
        print(f" VISION RETRIEVAL WITH CHOICES")
        print(f"Query Image: {Path(sample_image).name}")
        print("-" * 50)
        
        # Display the query image first
        print(f"\n🔍 QUERY IMAGE:")
        try:
            from IPython.display import display, Image as IPImage
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
            
            if Path(sample_image).exists():
                try:
                    display(IPImage(filename=sample_image, width=400, height=300))
                    print(f"    ✅ Query image displayed above: {Path(sample_image).name}")
                except:
                    img = mpimg.imread(sample_image)
                    plt.figure(figsize=(8, 6))
                    plt.imshow(img)
                    plt.title(f"Query Image: {Path(sample_image).name}")
                    plt.axis('off')
                    plt.tight_layout()
                    plt.show()
                    print(f"    ✅ Query image displayed above using matplotlib")
            else:
                print(f"    ❌ Query image not found: {sample_image}")
        except Exception as e:
            print(f"    ❌ Could not display query image: {e}")
        
        print(f"\n📋 RETRIEVAL RESULTS:")
        
        # Create query structure for VISION ONLY
        query_structure = QueryStructure(
            strategy_name=f"vision_question_{question_idx}",
            modalities=["vision"],  # ONLY VISION
            image_path=sample_image,
            description=f"Vision retrieval for question {question_idx}"
        )
        
        print(" Processing vision-only query...")
        print("   (Watch for 'Processing VISION ONLY' message)")
        
        # Perform retrieval using FIXED functions
        distances, indices = retrieve_with_modality(query_structure, k=top_k)
        retrieved_video_ids = [self.data_entries[i]['video_id'] for i in indices]
        
        print(f"\n Top {top_k} Vision Retrieval Results:")
        
        # Import display libraries for notebook
        try:
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
            from IPython.display import display, Image as IPImage, HTML
            import PIL.Image
            import base64
            from io import BytesIO
            
            # Enable inline plotting
            plt.rcParams['figure.facecolor'] = 'white'
            
            for rank, (video_id, distance) in enumerate(zip(retrieved_video_ids, distances), 1):
                video_entry = next(e for e in self.data_entries if e['video_id'] == video_id)
                
                print(f"\n Rank {rank} (Score: {distance:.3f})")
                print(f" Video: {video_id}")
                print(f" Image: {Path(video_entry['image_path']).name}")
                
                # Display the actual image
                image_path = video_entry['image_path']
                if Path(image_path).exists():
                    try:
                        # Method 1: Try IPython Image display first
                        try:
                            display(IPImage(filename=image_path, width=400, height=300))
                            print(f"    ✅ Image displayed above using IPython.display.Image")
                        except:
                            # Method 2: Try matplotlib
                            img = mpimg.imread(image_path)
                            plt.figure(figsize=(8, 6))
                            plt.imshow(img)
                            plt.title(f"Rank {rank}: Video {video_id} (Score: {distance:.3f})")
                            plt.axis('off')
                            plt.tight_layout()
                            plt.show()
                            print(f"    ✅ Image displayed above using matplotlib")
                        
                        # Show image info
                        try:
                            with PIL.Image.open(image_path) as pil_img:
                                print(f"    📐 Dimensions: {pil_img.size[0]}x{pil_img.size[1]} pixels")
                                print(f"    🎨 Mode: {pil_img.mode}")
                        except:
                            print(f"    📐 Could not read image dimensions")
                        
                        # Show file size
                        file_size = Path(image_path).stat().st_size
                        print(f"    📁 Size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
                        
                    except Exception as e:
                        print(f"    ❌ Could not display image: {e}")
                        print(f"    📂 Path: {image_path}")
                        # Try to show if file exists and is readable
                        if Path(image_path).exists():
                            print(f"    ℹ️  File exists but cannot be displayed")
                        else:
                            print(f"    ⚠️  File does not exist")
                else:
                    print(f"    ❌ Image file not found: {image_path}")
            
        except ImportError as e:
            print(f" ❌ Could not import display libraries: {e}")
            print(" 📋 Showing image paths instead:")
            
            for rank, (video_id, distance) in enumerate(zip(retrieved_video_ids, distances), 1):
                video_entry = next(e for e in self.data_entries if e['video_id'] == video_id)
                print(f"\n Rank {rank} (Score: {distance:.3f})")
                print(f" Video: {video_id}")
                print(f" Image: {Path(video_entry['image_path']).name}")
                print(f" Path: {video_entry['image_path']}")
        
        print(f"\n✅ VISION-ONLY processing complete (no text/audio processed)")
    
    def show_audio_retrieval_with_choices(self, question_idx: int, top_k: int = 3, query_audio_path: str = None):
        """Show audio retrieval results - simplified to prevent hanging"""
        if not self.setup_complete:
            print(" Please run setup_showcase() first!")
            return
            
        if question_idx >= len(self.question_data['questions']):
            print(f" Question index {question_idx} out of range!")
            return
            
        # Use provided query audio or default to first video's audio
        if query_audio_path is None:
            sample_audio = self.data_entries[0]['audio_path']
        else:
            sample_audio = query_audio_path
        
        print(f" AUDIO RETRIEVAL WITH CHOICES")
        print(f"Query Audio: {Path(sample_audio).name}")
        print("-" * 50)
        
        # Show query audio with playback
        print(f"\n🔊 QUERY AUDIO:")
        if Path(sample_audio).exists():
            file_size = Path(sample_audio).stat().st_size
            print(f"    📁 File: {Path(sample_audio).name}")
            print(f"    📁 Size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
            
            # Add audio player with protection
            try:
                from IPython.display import Audio, display
                print(f"    🎧 Creating audio player...")
                audio_widget = Audio(filename=sample_audio, autoplay=False)
                display(audio_widget)
                print(f"    ✅ Query audio player displayed above")
            except Exception as e:
                print(f"    ❌ Audio player failed: {e}")
                print(f"    💡 Manual playback: vlc '{sample_audio}'")
        else:
            print(f"    ❌ Query audio not found: {sample_audio}")
        
        # Create query structure for AUDIO ONLY
        query_structure = QueryStructure(
            strategy_name=f"audio_question_{question_idx}",
            modalities=["audio"],  # ONLY AUDIO
            audio_path=sample_audio,
            description=f"Audio retrieval for question {question_idx}"
        )
        
        print(" Processing audio-only query...")
        
        # Perform retrieval using FIXED functions
        distances, indices = retrieve_with_modality(query_structure, k=top_k)
        retrieved_video_ids = [self.data_entries[i]['video_id'] for i in indices]
        
        print(f"\n Top {top_k} Audio Retrieval Results:")
        
        # Show results with basic info only (no audio loading/processing)
        for rank, (video_id, distance) in enumerate(zip(retrieved_video_ids, distances), 1):
            video_entry = next(e for e in self.data_entries if e['video_id'] == video_id)
            
            print(f"\n Rank {rank} (Score: {distance:.3f})")
            print(f" Video: {video_id}")
            print(f" Audio: {Path(video_entry['audio_path']).name}")
            
            # Show basic file info and add audio player
            audio_path = video_entry['audio_path']
            if Path(audio_path).exists():
                file_size = Path(audio_path).stat().st_size
                print(f"    📁 Size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
                
                # Add audio player with protection
                try:
                    from IPython.display import Audio, display
                    #import asyncio
                    print(f"    🎧 Creating audio player...")
                    audio_widget = Audio(filename=audio_path, autoplay=False)
                    #asyncio.create_task(display(audio_widget))
                    print(f"    ✅ Audio player displayed above")
                except Exception as e:
                    print(f"    ❌ Audio player failed: {e}")
                    print(f"    💡 Manual playback: vlc '{audio_path}'")
            else:
                print(f"    ❌ Audio file not found: {audio_path}")
        
        print(f"\n✅ AUDIO-ONLY processing complete (no text/vision processed)")
        print("💡 Audio files shown with basic info only to prevent hanging")
    
    def show_multimodal_retrieval_with_choices(self, question_idx: int, top_k: int = 3, 
                                         query_image_path: str = None, 
                                         query_audio_path: str = None):
        """Show multimodal retrieval results with all content types displayed"""
        if not self.setup_complete:
            print(" Please run setup_showcase() first!")
            return
            
        if question_idx >= len(self.question_data['questions']):
            print(f" Question index {question_idx} out of range!")
            return
            
        # Get question text
        question_text = self.question_data['questions'].iloc[question_idx]
        sample_image = query_image_path if query_image_path else self.data_entries[0]['image_path']
        sample_audio = query_audio_path if query_audio_path else self.data_entries[0]['audio_path']
        
        print(f" MULTIMODAL RETRIEVAL WITH CHOICES")
        print(f"Question: {question_text}")
        if query_image_path:
            print(f"🖼️  Using custom query image: {Path(query_image_path).name}")
        else:
            print(f"Sample Image: {Path(sample_image).name}")
        if query_audio_path:
            print(f"🔊 Using custom query audio: {Path(query_audio_path).name}")
        else:
            print(f"Sample Audio: {Path(sample_audio).name}")
        print("-" * 50)
        
        # Create query structure for ALL MODALITIES
        query_structure = QueryStructure(
            strategy_name=f"multimodal_question_{question_idx}",
            modalities=["text", "vision", "audio"],  # ALL MODALITIES
            text_query=question_text,
            image_path=sample_image,
            audio_path=sample_audio,
            description=f"Multimodal retrieval for question {question_idx}"
        )
        
        print(" Processing multimodal query...")
        print("   (Watch for 'Processing ALL MODALITIES' message)")
        
        # Perform retrieval using FIXED functions
        distances, indices = retrieve_with_modality(query_structure, k=top_k)
        retrieved_video_ids = [self.data_entries[i]['video_id'] for i in indices]
        
        print(f"\n Top {top_k} Multimodal Retrieval Results:")
        
        # Import display libraries
        try:
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
            from IPython.display import Audio, display, Image as IPImage, HTML
            import PIL.Image
            import librosa
            
            # Enable inline plotting
            plt.rcParams['figure.facecolor'] = 'white'
            
            for rank, (video_id, distance) in enumerate(zip(retrieved_video_ids, distances), 1):
                video_entry = next(e for e in self.data_entries if e['video_id'] == video_id)
                
                print(f"\n{'='*20} Rank {rank} (Score: {distance:.3f}) {'='*20}")
                print(f"🎬 Video: {video_id}")
                print("=" * 60)
                
                # 1. Show FULL TEXT content
                print(f"📝 TEXT CONTENT:")
                text_content = video_entry.get('text_content', video_entry.get('text', 'N/A'))
                print(f"   {text_content}")
                print("-" * 40)
                
                # 2. Show IMAGE
                print(f"🖼️  IMAGE CONTENT:")
                image_path = video_entry['image_path']
                if Path(image_path).exists():
                    try:
                        # Try multiple display methods for images
                        try:
                            # Method 1: IPython Image display
                            display(IPImage(filename=image_path, width=400, height=300))
                            print(f"    ✅ Image displayed above (IPython.display.Image)")
                        except:
                            # Method 2: matplotlib
                            img = mpimg.imread(image_path)
                            plt.figure(figsize=(8, 6))
                            plt.imshow(img)
                            plt.title(f"Rank {rank}: Video {video_id} - Image")
                            plt.axis('off')
                            plt.tight_layout()
                            plt.show()
                            print(f"    ✅ Image displayed above (matplotlib)")
                        
                        # Show image info
                        try:
                            with PIL.Image.open(image_path) as pil_img:
                                print(f"    📐 Dimensions: {pil_img.size[0]}x{pil_img.size[1]} pixels")
                                print(f"    🎨 Mode: {pil_img.mode}")
                        except:
                            pass
                        
                        file_size = Path(image_path).stat().st_size
                        print(f"    📁 Size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
                        
                    except Exception as e:
                        print(f"    ❌ Could not display image: {e}")
                        print(f"    📂 Path: {image_path}")
                else:
                    print(f"    ❌ Image file not found: {image_path}")
                print("-" * 40)
                
                # 3. Show AUDIO
                print(f"🎵 AUDIO CONTENT:")
                audio_path = video_entry['audio_path']
                if Path(audio_path).exists():
                    try:
                        # Show audio info first
                        file_size = Path(audio_path).stat().st_size
                        print(f"    📁 Size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
                        
                        # Try to get audio info
                        try:
                            y, sr = librosa.load(audio_path, sr=None)
                            duration = len(y) / sr
                            print(f"    ⏱ Duration: {duration:.2f} seconds")
                            print(f"    🎵 Sample Rate: {sr} Hz")
                        except:
                            print(f"    ⏱ Duration: Could not determine")
                        
                        # Create audio player - try multiple methods
                        print(f"    🎧 Audio Player:")
                        try:
                            # Method 1: Direct IPython Audio
                            audio_widget = Audio(filename=audio_path, autoplay=False, embed=True)
                            #display(audio_widget)
                            print(f"    ✅ Audio player displayed above")
                        except Exception as e1:
                            try:
                                # Method 2: Try with data parameter
                                with open(audio_path, 'rb') as f:
                                    audio_data = f.read()
                                audio_widget = Audio(data=audio_data, autoplay=False, embed=True)
                                #display(audio_widget)
                                print(f"    ✅ Audio player displayed above (data method)")
                            except Exception as e2:
                                print(f"    ❌ Could not create audio player: {e1}")
                                print(f"    💡 Try manually: vlc '{audio_path}' or mpv '{audio_path}'")
                        
                    except Exception as e:
                        print(f"    ❌ Could not process audio: {e}")
                        print(f"    📂 Path: {audio_path}")
                else:
                    print(f"    ❌ Audio file not found: {audio_path}")
                
                print("=" * 60)
            
        except ImportError as e:
            print(f" ❌ Could not import display libraries: {e}")
            print(" 📋 Showing content info instead:")
            
            for rank, (video_id, distance) in enumerate(zip(retrieved_video_ids, distances), 1):
                video_entry = next(e for e in self.data_entries if e['video_id'] == video_id)
                
                print(f"\n Rank {rank} (Score: {distance:.3f})")
                print(f" Video: {video_id}")
                
                # Show all modalities info
                text_content = video_entry.get('text_content', video_entry.get('text', 'N/A'))
                print(f" 📝 Text: {text_content}")
                print(f" 🖼️  Image: {Path(video_entry['image_path']).name}")
                print(f" 🎵 Audio: {Path(video_entry['audio_path']).name}")
                
                # Show file existence
                image_exists = "✅" if Path(video_entry['image_path']).exists() else "❌"
                audio_exists = "✅" if Path(video_entry['audio_path']).exists() else "❌"
                print(f" Files: Image {image_exists} | Audio {audio_exists}")
        
        print(f"\n✅ MULTIMODAL processing complete (all modalities processed)")
    
    def demo_questions_showcase(self, demo_questions: list = [0, 1, 2]):
        """
        Demo multiple questions with all modalities - supports your exact usage pattern
        
        Usage:
        showcase = MultimodalShowcase()
        showcase.setup_showcase()
        
        demo_questions = [0, 1, 2]
        for question_idx in demo_questions:
            question = showcase.question_data['questions'].iloc[question_idx]
            movie_name = showcase.question_data['dataframe'].iloc[question_idx]['movie_name']
            
            print(f"\\n" + "" * 25)
            print(f"DEMO QUESTION {question_idx}: {movie_name}")
            print(f"Question: {question}")
            print("" * 50)
            
            print(f"\\n" + "" * 15 + " TEXT WITH CHOICES " + "" * 15)
            showcase.show_text_retrieval_with_choices(question_idx, top_k=3)
        """
        if not self.setup_complete:
            print(" Please run setup_showcase() first!")
            return
            
        print(f" DEMO: MULTIPLE QUESTIONS WITH FIXED MODALITY PROCESSING")
        print("=" * 70)
        
        for question_idx in demo_questions:
            if question_idx >= len(self.question_data['questions']):
                print(f" Question index {question_idx} out of range!")
                continue
                
            question = self.question_data['questions'].iloc[question_idx]
            movie_name = self.question_data['dataframe'].iloc[question_idx]['movie_name']
            
            print(f"\n" + "" * 25)
            print(f"DEMO QUESTION {question_idx}: {movie_name}")
            print(f"Question: {question}")
            print("" * 50)
            
            # Demo 1: Text retrieval with choices
            print(f"\n" + "" * 15 + " TEXT WITH CHOICES " + "" * 15)
            self.show_text_retrieval_with_choices(question_idx, top_k=3)
            
            # Demo 2: Vision retrieval
            print(f"\n" + "" * 15 + " VISION RETRIEVAL " + "" * 15)
            self.show_vision_retrieval_with_choices(question_idx, top_k=3)
            
            # Demo 3: Audio retrieval
            print(f"\n" + "" * 15 + " AUDIO RETRIEVAL " + "" * 15)
            self.show_audio_retrieval_with_choices(question_idx, top_k=3)
            
            # Demo 4: Multimodal retrieval
            print(f"\n" + "" * 15 + " MULTIMODAL RETRIEVAL " + "" * 15)
            self.show_multimodal_retrieval_with_choices(question_idx, top_k=3)
            
            print(f"\n COMPLETED DEMO FOR QUESTION {question_idx}")
            print("=" * 70)

print(" MultimodalShowcase class added to utils.py")
print(" Usage with existing variables:")
print("   showcase = MultimodalShowcase(data_entries, question_data, text_index, vision_index, audio_index, multimodal_indices, normalized_embeddings)")
print("   showcase.setup_showcase()")
print(" Or let it use global variables:")
print("   showcase = MultimodalShowcase()")
print("   showcase.setup_showcase()")

# =============================================================================
# ENHANCED MULTIMODAL EVALUATION CLASS
# =============================================================================

class EnhancedMultimodalEvaluator:
    """
    Comprehensive evaluation class that runs text, audio, visual, and multimodal evaluations
    with both traditional IR metrics and enhanced RAGAS metrics
    """
    
    def __init__(self, sagemaker_endpoint_name: str, num_questions_to_test: int = 5):
        """
        Initialize the evaluator
        
        Args:
            sagemaker_endpoint_name: Name of the SageMaker endpoint for LLM
            num_questions_to_test: Number of questions to evaluate
        """
        self.sagemaker_endpoint_name = sagemaker_endpoint_name
        self.num_questions_to_test = num_questions_to_test
        
        # Components that will be initialized
        self.llm = None
        self.embeddings = None
        self.custom_metrics = None
        self.ragas_metrics = None
        
        # Data components
        self.data_entries = None
        self.question_data = None
        
        # Results storage
        self.evaluation_results = []
        
    def setup_evaluation_components(self):
        """Setup all components needed for evaluation"""
        print(f"\n Setting up Enhanced Multimodal Evaluation Components...")
        print(f"   SageMaker Endpoint: {self.sagemaker_endpoint_name}")
        print(f"   Questions to test: {self.num_questions_to_test}")
        
        # Load data if not already loaded
        if self.data_entries is None:
            print(" Loading data entries...")
            self.data_entries = load_cinepile_data()
            globals()['data_entries'] = self.data_entries
            
        if self.question_data is None:
            print(" Loading question data...")
            self.question_data = load_cinepile_questions()
            
        # Setup SageMaker LLM
        print(" Setting up SageMaker LLM...")
        sm = boto3.Session().client('sagemaker-runtime')
        chat_content_handler = ContentHandler()
        
        self.llm = ChatSagemakerEndpoint(
            name="Qwen2.5-Enhanced-RAGAS",
            endpoint_name=self.sagemaker_endpoint_name,
            client=sm,
            model_kwargs={
                "temperature": 0.7,
                "max_new_tokens": 1200,
                "top_p": 0.95,
                "do_sample": True
            },
            content_handler=chat_content_handler
        )
        
        # Setup Enhanced ImageBind embeddings
        print(" Setting up Enhanced ImageBind embeddings...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Use global model if available, otherwise load
        if 'model' in globals():
            imagebind_model_instance = globals()['model']
            print("   Using existing global ImageBind model")
        else:
            print("   Loading ImageBind model...")
            imagebind_model_instance = imagebind_model.imagebind_huge(pretrained=True).eval().to(device)
        
        self.embeddings = EnhancedMultimodalImageBindEmbeddings(
            model=imagebind_model_instance,
            device=device,
            data_entries=self.data_entries
        )
        
        # Setup custom multimodal metrics
        print(" Setting up custom multimodal metrics...")
        self.custom_metrics = CustomMultimodalMetrics(self.embeddings)
        
        # Define RAGAS metrics (text-based)
        print(" Setting up RAGAS metrics...")
        self.ragas_metrics = [
            answer_relevancy,
            faithfulness,
            context_precision,
            context_recall,
        ]
        
        # Initialize RAGAS metrics
        self._init_ragas_metrics()
        
        print(" Enhanced multimodal evaluation components setup complete")
        
    def _init_ragas_metrics(self):
        """Initialize RAGAS metrics with enhanced embeddings"""
        print(" Initializing RAGAS metrics with enhanced multimodal embeddings...")
        
        for metric in self.ragas_metrics:
            if isinstance(metric, MetricWithLLM):
                print(f"  {metric.name} <- LLM")
                metric.llm = LangchainLLMWrapper(self.llm)
            if isinstance(metric, MetricWithEmbeddings):
                print(f"  {metric.name} <- Enhanced Multimodal ImageBind Embeddings")
                metric.embeddings = LangchainEmbeddingsWrapper(self.embeddings)
            run_config = RunConfig()
            metric.init(run_config)
        
        print(" RAGAS metrics initialized with enhanced embeddings")
    
    async def _score_with_enhanced_ragas(self, query, chunks, answer, query_structure, 
                                       retrieved_video_ids, expected_video_id):
        """
        Score using both RAGAS (for text) and custom multimodal metrics
        """
        scores = {}
        
        # 1. Traditional RAGAS metrics (text-based)
        print("     Calculating traditional RAGAS metrics...")
        for metric in self.ragas_metrics:
            sample = SingleTurnSample(
                user_input=query,
                retrieved_contexts=chunks,
                response=answer,
                reference=chunks[0] if chunks else ""
            )
            try:
                scores[f"ragas_{metric.name}"] = await metric.single_turn_ascore(sample)
                print(f"     ragas_{metric.name}: {scores[f'ragas_{metric.name}']:.3f}")
            except Exception as e:
                print(f"     Error calculating ragas_{metric.name}: {e}")
                scores[f"ragas_{metric.name}"] = 0.0
        
        # 2. Custom multimodal metrics
        print("     Calculating custom multimodal metrics...")
        try:
            # Multimodal similarity metrics
            similarity_metrics = self.custom_metrics.calculate_multimodal_similarity(
                query_structure, retrieved_video_ids, expected_video_id
            )
            scores.update({f"multimodal_{k}": v for k, v in similarity_metrics.items() if k != 'similarities'})
            
            # Multimodal faithfulness
            multimodal_faithfulness = self.custom_metrics.calculate_multimodal_faithfulness(
                query_structure, retrieved_video_ids, expected_video_id, answer
            )
            scores['multimodal_faithfulness'] = multimodal_faithfulness
            
            # Multimodal relevancy
            multimodal_relevancy = self.custom_metrics.calculate_multimodal_relevancy(
                query_structure, retrieved_video_ids
            )
            scores['multimodal_relevancy'] = multimodal_relevancy
            
            print(f"     multimodal_faithfulness: {multimodal_faithfulness:.3f}")
            print(f"     multimodal_relevancy: {multimodal_relevancy:.3f}")
            print(f"     multimodal_avg_similarity: {similarity_metrics['avg_similarity']:.3f}")
            
        except Exception as e:
            print(f"     Error calculating multimodal metrics: {e}")
            scores.update({
                'multimodal_faithfulness': 0.0,
                'multimodal_relevancy': 0.0,
                'multimodal_avg_similarity': 0.0,
                'multimodal_max_similarity': 0.0,
                'multimodal_expected_similarity': 0.0
            })
        
        return scores
    
    def run_comprehensive_evaluation(self):
        """Run the complete enhanced multimodal evaluation"""
        print(f"\n RUNNING COMPREHENSIVE MULTIMODAL EVALUATION")
        print("=" * 60)
        
        # Setup components if not already done
        if self.llm is None:
            self.setup_evaluation_components()
        
        # Get questions data
        questions = self.question_data['questions']
        choices = self.question_data['choices']
        answer_keys = self.question_data['answer_keys']
        dataframe = self.question_data['dataframe']
        
        # Results storage
        self.evaluation_results = []
        
        # Test questions
        num_questions = min(self.num_questions_to_test, len(questions))
        print(f" Testing {num_questions} questions with enhanced multimodal evaluation...")
        
        for idx in range(num_questions):
            question = questions.iloc[idx]
            choice_list = choices.iloc[idx]
            answer_key = answer_keys.iloc[idx]
            expected_video_id = idx
            movie_name = dataframe.iloc[idx]['movie_name']
            
            print(f"\n Question {idx+1}/{num_questions}: {movie_name}")
            print(f"   Q: {question}")
            
            # Get reference video data
            video_entries = [entry for entry in self.data_entries if entry['video_id'] == expected_video_id]
            if not video_entries:
                continue
            
            reference_entry = video_entries[0]
            question_with_choices = f"{question} Answer choices: {', '.join(choice_list)}"
            
            # Test different strategies with appropriate modalities
            query_strategies = [
                QueryStructure(
                    strategy_name="text_only",
                    modalities=["text"],
                    text_query=question_with_choices,
                    description="Uses question + choices text for retrieval"
                ),
                QueryStructure(
                    strategy_name="vision_only",
                    modalities=["vision"],
                    image_path=reference_entry['image_path'],
                    description="Uses only visual content for retrieval"
                ),
                QueryStructure(
                    strategy_name="audio_only",
                    modalities=["audio"],
                    audio_path=reference_entry['audio_path'],
                    description="Uses only audio content for retrieval"
                ),
                QueryStructure(
                    strategy_name="multimodal_full",
                    modalities=["text", "vision", "audio"],
                    text_query=question_with_choices,
                    image_path=reference_entry['image_path'],
                    audio_path=reference_entry['audio_path'],
                    description="Uses all modalities"
                )
            ]
            
            question_results = {
                'question_idx': idx,
                'movie_name': movie_name,
                'question': question,
                'expected_video': expected_video_id,
                'answer_key': answer_key,
                'choices': choice_list,
                'strategies': {}
            }
            
            for query_structure in query_strategies:
                try:
                    print(f"\n    Testing {query_structure.strategy_name} with {query_structure.modalities}...")
                    
                    # Perform retrieval
                    distances, indices = retrieve_with_modality(query_structure, k=10)
                    retrieved_video_ids = [self.data_entries[i]['video_id'] for i in indices]
                    
                    # Calculate standard IR metrics
                    standard_metrics = calculate_comprehensive_metrics(retrieved_video_ids, expected_video_id, distances)
                    
                    # Get retrieved contexts for RAGAS (text-based)
                    retrieved_contexts = []
                    for video_id in retrieved_video_ids[:3]:
                        video_entries_for_context = [e for e in self.data_entries if e['video_id'] == video_id]
                        if video_entries_for_context:
                            retrieved_contexts.append(video_entries_for_context[0]['text'])
                        else:
                            retrieved_contexts.append("No content available")
                    
                    # Calculate enhanced metrics (RAGAS + Custom Multimodal)
                    print(f"    Calculating enhanced metrics for {query_structure.strategy_name}...")
                    try:
                        import asyncio
                        enhanced_scores = asyncio.run(self._score_with_enhanced_ragas(
                            question, retrieved_contexts, answer_key, 
                            query_structure, retrieved_video_ids, expected_video_id
                        ))
                    except Exception as e:
                        print(f"    Enhanced scoring failed: {e}")
                        enhanced_scores = {}
                    
                    # Store results
                    strategy_results = {
                        'query_structure': query_structure.get_summary(),
                        'retrieved_videos': retrieved_video_ids[:5],
                        'distances': distances[:5],
                        'standard_metrics': standard_metrics,
                        'enhanced_scores': enhanced_scores
                    }
                    
                    question_results['strategies'][query_structure.strategy_name] = strategy_results
                    
                    # Print summary
                    top1_correct = "" if retrieved_video_ids[0] == expected_video_id else ""
                    print(f"    {query_structure.strategy_name}: {top1_correct} Top-1")
                    print(f"       Standard: P@1={standard_metrics['precision@1']:.3f}, MRR@5={standard_metrics['mrr@5']:.3f}")
                    if enhanced_scores:
                        print(f"       Enhanced: Multimodal Faithfulness={enhanced_scores.get('multimodal_faithfulness', 0):.3f}")
                    
                except Exception as e:
                    print(f"    {query_structure.strategy_name}: Error - {e}")
                    import traceback
                    traceback.print_exc()
            
            self.evaluation_results.append(question_results)
        
        return self.evaluation_results
    
    def create_comparison_table(self):
        """Create a comprehensive comparison table with highlighted best metrics"""
        if not self.evaluation_results:
            print(" No evaluation results available. Run evaluation first.")
            return None
        
        print(f"\n CREATING COMPREHENSIVE COMPARISON TABLE")
        print("=" * 60)
        
        # Aggregate results by strategy
        strategy_metrics = {}
        
        for result in self.evaluation_results:
            for strategy_name, strategy_data in result['strategies'].items():
                if strategy_name not in strategy_metrics:
                    strategy_metrics[strategy_name] = {
                        'standard_metrics': {},
                        'enhanced_scores': {},
                        'count': 0
                    }
                
                # Aggregate standard metrics
                for metric_name, value in strategy_data['standard_metrics'].items():
                    if metric_name not in strategy_metrics[strategy_name]['standard_metrics']:
                        strategy_metrics[strategy_name]['standard_metrics'][metric_name] = []
                    strategy_metrics[strategy_name]['standard_metrics'][metric_name].append(value)
                
                # Aggregate enhanced scores
                for metric_name, value in strategy_data['enhanced_scores'].items():
                    if metric_name not in strategy_metrics[strategy_name]['enhanced_scores']:
                        strategy_metrics[strategy_name]['enhanced_scores'][metric_name] = []
                    strategy_metrics[strategy_name]['enhanced_scores'][metric_name].append(value)
                
                strategy_metrics[strategy_name]['count'] += 1
        
        # Calculate averages
        averaged_metrics = {}
        for strategy_name, metrics in strategy_metrics.items():
            averaged_metrics[strategy_name] = {}
            
            # Average standard metrics
            for metric_name, values in metrics['standard_metrics'].items():
                averaged_metrics[strategy_name][f"standard_{metric_name}"] = np.mean(values)
            
            # Average enhanced scores
            for metric_name, values in metrics['enhanced_scores'].items():
                averaged_metrics[strategy_name][f"enhanced_{metric_name}"] = np.mean(values)
        
        # Create DataFrame for better visualization
        import pandas as pd
        
        # Prepare data for DataFrame
        table_data = []
        
        # Define key metrics to include in table
        key_metrics = [
            'standard_precision@1', 'standard_recall@1', 'standard_mrr@5', 'standard_ndcg@5',
            'enhanced_ragas_answer_relevancy', 'enhanced_ragas_faithfulness', 
            'enhanced_ragas_context_precision', 'enhanced_ragas_context_recall',
            'enhanced_multimodal_faithfulness', 'enhanced_multimodal_relevancy', 
            'enhanced_multimodal_avg_similarity'
        ]
        
        for strategy_name in ['text_only', 'vision_only', 'audio_only', 'multimodal_full']:
            if strategy_name in averaged_metrics:
                row = {'Strategy': strategy_name.replace('_', ' ').title()}
                for metric in key_metrics:
                    if metric in averaged_metrics[strategy_name]:
                        row[metric.replace('standard_', '').replace('enhanced_', '').replace('_', ' ').title()] = averaged_metrics[strategy_name][metric]
                    else:
                        row[metric.replace('standard_', '').replace('enhanced_', '').replace('_', ' ').title()] = 0.0
                table_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(table_data)
        df = df.set_index('Strategy')
        
        # Find best values for each metric (for highlighting)
        best_values = {}
        for col in df.columns:
            best_values[col] = df[col].max()
        
        print(" COMPREHENSIVE EVALUATION RESULTS TABLE")
        print("=" * 100)
        
        # Display table with formatting
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.float_format', '{:.3f}'.format)
        
        print(df.to_string())
        
        # Highlight best performers
        print(f"\n BEST PERFORMING METRICS (Highlighted in Green):")
        print("=" * 60)
        
        for col in df.columns:
            best_strategy = df[col].idxmax()
            best_value = df[col].max()
            print(f" {col}: {best_strategy} ({best_value:.3f})")
        
        # Create modality effectiveness analysis
        print(f"\n MODALITY EFFECTIVENESS ANALYSIS:")
        print("=" * 60)
        
        modality_analysis = {
            'Text Only': df.loc['Text Only'].mean(),
            'Vision Only': df.loc['Vision Only'].mean(),
            'Audio Only': df.loc['Audio Only'].mean(),
            'Multimodal Full': df.loc['Multimodal Full'].mean()
        }
        
        sorted_modalities = sorted(modality_analysis.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (modality, avg_score) in enumerate(sorted_modalities, 1):
            print(f"{rank}. {modality}: {avg_score:.3f} (average across all metrics)")
        
        return df, best_values, averaged_metrics
    
    def create_styled_comparison_table(self):
        """Create a styled comparison table with green highlighting for best metrics"""
        df, best_values, averaged_metrics = self.create_comparison_table()
        
        if df is None:
            return None
        
        try:
            # Create styled table with conditional formatting
            def highlight_best(val, best_val):
                if abs(val - best_val) < 0.001:  # Account for floating point precision
                    return 'background-color: lightgreen; font-weight: bold; color: black'
                return ''
            
            # Apply styling
            styled_df = df.style.apply(
                lambda col: [highlight_best(val, best_values[col.name]) for val in col], 
                axis=0
            )
            
            # Set table properties
            styled_df = styled_df.set_properties(**{
                'text-align': 'center',
                'border': '1px solid black',
                'border-collapse': 'collapse'
            })
            
            # Set caption
            styled_df = styled_df.set_caption(
                "Comprehensive Multimodal Evaluation Results - Best Values Highlighted in Green"
            )
            
            print(f"\n STYLED COMPARISON TABLE CREATED")
            print("   Best performing metrics are highlighted in green with bold text")
            
            return styled_df
            
        except Exception as e:
            print(f" Could not create styled table: {e}")
            print("   Returning basic DataFrame instead")
            return df
    
    def save_results(self, filename: str = "enhanced_multimodal_evaluation_results.json"):
        """Save evaluation results to JSON file"""
        if not self.evaluation_results:
            print(" No evaluation results to save")
            return
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.evaluation_results, f, indent=2, default=str)
            print(f" Results saved to {filename}")
        except Exception as e:
            print(f" Error saving results: {e}")
    
    def load_results(self, filename: str = "enhanced_multimodal_evaluation_results.json"):
        """Load evaluation results from JSON file"""
        try:
            with open(filename, 'r') as f:
                self.evaluation_results = json.load(f)
            print(f" Results loaded from {filename}")
            return self.evaluation_results
        except Exception as e:
            print(f" Error loading results: {e}")
            return None

print(" EnhancedMultimodalEvaluator class added to utils.py")
print(" Usage:")
print("   evaluator = EnhancedMultimodalEvaluator('your-sagemaker-endpoint', num_questions_to_test=5)")
print("   results = evaluator.run_comprehensive_evaluation()")
print("   styled_table = evaluator.create_styled_comparison_table()")
print("   evaluator.save_results()")
