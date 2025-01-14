import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
import spacy
from typing import List, Tuple, Dict
from collections import defaultdict
from dataclasses import dataclass
from scipy.spatial.distance import cosine

@dataclass
class TextEmbedding:
    text: str
    embedding: np.ndarray
    metadata: Dict = None

class FAISSTextAnalyzer:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize FAISS index
        self.embedding_dim = self.model.config.hidden_size
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Enable GPU if available
        if torch.cuda.is_available():
            self.res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(self.res, 0, self.index)
        
        # Store text embeddings for retrieval
        self.stored_embeddings: List[TextEmbedding] = []

    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1)

        return embedding.cpu().numpy()

    def add_texts(self, texts: List[str], metadata: List[Dict] = None):
        """Add multiple texts to the FAISS index."""
        if metadata is None:
            metadata = [None] * len(texts)
            
        embeddings = []
        for text, meta in zip(texts, metadata):
            embedding = self.get_embedding(text)
            self.stored_embeddings.append(TextEmbedding(text=text, embedding=embedding, metadata=meta))
            embeddings.append(embedding)
            
        embeddings = np.vstack(embeddings)
        self.index.add(embeddings)

    def search(self, query_text: str, k: int = 5) -> List[Tuple[float, TextEmbedding]]:
        """Search for similar texts and return distances and text objects."""
        query_emb = self.get_embedding(query_text)
        D, I = self.index.search(query_emb, k)
        
        results = []
        for dist, idx in zip(D[0], I[0]):
            results.append((dist, self.stored_embeddings[idx]))
            
        return results

    def syntactic_features(self, text: str) -> Dict[str, float]:
        """Extract syntactic features using spaCy."""
        doc = self.nlp(text)
        features = defaultdict(int)

        # Analyze sentence structure
        for sent in doc.sents:
            features['sent_length'] += len(sent)
            features['dep_tree_depth'] += self._get_dep_tree_depth(sent.root)

        # POS tag distribution
        for token in doc:
            features[f'pos_{token.pos_}'] += 1

        # Normalize features
        total_tokens = len(doc)
        return {k: v/total_tokens for k, v in features.items()}

    def _get_dep_tree_depth(self, root) -> int:
        """Calculate dependency tree depth recursively."""
        if not list(root.children):
            return 0
        return 1 + max(self._get_dep_tree_depth(child) for child in root.children)

    def compare_texts(self, original: str, generated_samples: List[str]) -> Dict[str, List[float]]:
        """Compare original text against multiple generated samples using FAISS."""
        results = {
            'semantic_similarity': [],
            'syntactic_similarity': [],
            'combined_score': []
        }

        # Get features for original text
        orig_embedding = self.get_embedding(original)
        orig_syntactic = self.syntactic_features(original)

        # Add generated samples to FAISS index
        self.index.reset()  # Clear existing index
        sample_embeddings = []
        for sample in generated_samples:
            emb = self.get_embedding(sample)
            sample_embeddings.append(emb)
        sample_embeddings = np.vstack(sample_embeddings)
        self.index.add(sample_embeddings)

        # Get nearest neighbors for original text
        D, I = self.index.search(orig_embedding, len(generated_samples))
        
        for i, sample in enumerate(generated_samples):
            # Semantic similarity from FAISS distance
            semantic_sim = 1.0 / (1.0 + D[0][i])  # Convert distance to similarity score
            
            # Syntactic similarity using feature vectors
            sample_syntactic = self.syntactic_features(sample)
            syntactic_sim = 1.0 - cosine(
                np.array(list(orig_syntactic.values())), 
                np.array(list(sample_syntactic.values()))
            )

            # Combined score (weighted average)
            combined = 0.6 * semantic_sim + 0.4 * syntactic_sim

            results['semantic_similarity'].append(semantic_sim)
            results['syntactic_similarity'].append(syntactic_sim)
            results['combined_score'].append(combined)

        return results

    def batch_compare(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Efficiently compare multiple texts in batches."""
        n = len(texts)
        similarity_matrix = np.zeros((n, n))
        
        # Process in batches
        for i in range(0, n, batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = []
            
            for text in batch_texts:
                emb = self.get_embedding(text)
                batch_embeddings.append(emb)
                
            batch_embeddings = np.vstack(batch_embeddings)
            
            # Compare batch against all texts
            D, _ = self.index.search(batch_embeddings, n)
            similarity_matrix[i:i + len(batch_texts)] = 1.0 / (1.0 + D)
            
        return similarity_matrix 