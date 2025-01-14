import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModel, GPT2LMHeadModel, GPT2Tokenizer
import spacy
from typing import List, Tuple, Dict, Any
from collections import defaultdict, Counter
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import wasserstein_distance, entropy
from scipy.spatial.distance import cosine
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.util import ngrams
from faiss_analyzer import FAISSTextAnalyzer
import torch.nn.functional as F

class EssayDetector:
    def __init__(self):
        # Initialize FAISS analyzer
        self.faiss_analyzer = FAISSTextAnalyzer()
        
        # Initialize language models
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2').to(self.faiss_analyzer.device)
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        
        # Initialize spaCy
        self.nlp = spacy.load('en_core_web_sm')

    def analyze_essay(self, 
                     student_essay: str, 
                     ai_essays: List[str],
                     plot: bool = True) -> Dict[str, Any]:
        """Comprehensive essay analysis using multiple methods."""
        results = {
            'faiss_analysis': self._faiss_analysis(student_essay, ai_essays),
            'statistical_analysis': self._statistical_analysis(student_essay, ai_essays),
            'mathematical_analysis': self._mathematical_analysis(student_essay, ai_essays),
            'nlp_analysis': self._nlp_analysis(student_essay, ai_essays)
        }
        
        if plot:
            self._plot_results(student_essay, ai_essays, results)
            
        return results

    def _faiss_analysis(self, student_essay: str, ai_essays: List[str]) -> Dict[str, Any]:
        """FAISS-based vector similarity analysis."""
        return self.faiss_analyzer.compare_texts(student_essay, ai_essays)

    def _statistical_analysis(self, student_essay: str, ai_essays: List[str]) -> Dict[str, float]:
        """Statistical analysis methods."""
        results = {}
        
        # Word frequency analysis
        student_freq = self._get_word_frequencies(student_essay)
        ai_freqs = [self._get_word_frequencies(essay) for essay in ai_essays]
        
        results['avg_freq_similarity'] = np.mean([
            1 - cosine(
                np.array(list(student_freq.values())), 
                np.array(list(ai_freq.values()))
            ) for ai_freq in ai_freqs
        ])
        
        # Perplexity analysis
        results['perplexity'] = self._calculate_perplexity(student_essay)
        results['avg_ai_perplexity'] = np.mean([
            self._calculate_perplexity(essay) for essay in ai_essays
        ])
        
        # Stylometry
        results.update(self._stylometry_analysis(student_essay, ai_essays))
        
        # Semantic coherence
        results['semantic_coherence'] = self._semantic_coherence(student_essay)
        results['avg_ai_coherence'] = np.mean([
            self._semantic_coherence(essay) for essay in ai_essays
        ])
        
        return results

    def _mathematical_analysis(self, student_essay: str, ai_essays: List[str]) -> Dict[str, float]:
        """Mathematical analysis methods."""
        results = {}
        
        # Total Variation distance
        student_dist = self._get_distribution(student_essay)
        ai_dists = [self._get_distribution(essay) for essay in ai_essays]
        
        results['tv_distances'] = [
            wasserstein_distance(student_dist, ai_dist)
            for ai_dist in ai_dists
        ]
        
        # Perturbation analysis
        results['perturbation_scores'] = self._perturbation_analysis(
            student_essay, ai_essays
        )
        
        return results

    def _nlp_analysis(self, student_essay: str, ai_essays: List[str]) -> Dict[str, Any]:
        """NLP-based analysis methods."""
        results = {}
        
        # Sentence structure analysis
        doc = self.nlp(student_essay)
        ai_docs = [self.nlp(essay) for essay in ai_essays]
        
        results['syntax_complexity'] = self._syntax_complexity(doc)
        results['avg_ai_complexity'] = np.mean([
            self._syntax_complexity(ai_doc) for ai_doc in ai_docs
        ])
        
        # N-gram analysis
        results['ngram_scores'] = self._ngram_analysis(student_essay, ai_essays)
        
        return results

    def _get_word_frequencies(self, text: str) -> Dict[str, float]:
        """Calculate normalized word frequencies."""
        words = word_tokenize(text.lower())
        freq = Counter(words)
        total = sum(freq.values())
        return {word: count/total for word, count in freq.items()}

    def _calculate_perplexity(self, text: str) -> float:
        """Calculate text perplexity using GPT-2."""
        encodings = self.gpt2_tokenizer(text, return_tensors='pt').to(self.faiss_analyzer.device)
        max_length = self.gpt2_model.config.max_position_embeddings
        stride = 512
        
        nlls = []
        for i in range(0, encodings.input_ids.size(1), stride):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, encodings.input_ids.size(1))
            trg_len = end_loc - i
            
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.faiss_analyzer.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            
            with torch.no_grad():
                outputs = self.gpt2_model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len
            
            nlls.append(neg_log_likelihood)
            
        return torch.exp(torch.stack(nlls).sum() / end_loc).item()

    def _stylometry_analysis(self, student_essay: str, ai_essays: List[str]) -> Dict[str, float]:
        """Analyze writing style features."""
        def get_style_features(text):
            doc = self.nlp(text)
            features = {
                'avg_sent_len': np.mean([len(sent) for sent in doc.sents]),
                'pos_ratios': defaultdict(float),
                'punct_ratio': len([t for t in doc if t.is_punct]) / len(doc),
                'function_word_ratio': len([t for t in doc if t.is_stop]) / len(doc)
            }
            for token in doc:
                features['pos_ratios'][token.pos_] += 1
            for pos in features['pos_ratios']:
                features['pos_ratios'][pos] /= len(doc)
            return features
        
        student_features = get_style_features(student_essay)
        ai_features = [get_style_features(essay) for essay in ai_essays]
        
        return {
            'style_similarity': np.mean([
                1 - np.mean([
                    abs(student_features[k] - ai_feature[k])
                    for k in student_features if k != 'pos_ratios'
                ]) for ai_feature in ai_features
            ])
        }

    def _semantic_coherence(self, text: str) -> float:
        """Calculate semantic coherence using embeddings."""
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return 1.0
            
        embeddings = []
        for sent in sentences:
            emb = self.faiss_analyzer.get_embedding(sent)
            embeddings.append(emb.flatten())
            
        coherence_scores = []
        for i in range(len(embeddings)-1):
            score = 1 - cosine(embeddings[i], embeddings[i+1])
            coherence_scores.append(score)
            
        return np.mean(coherence_scores)

    def _get_distribution(self, text: str) -> np.ndarray:
        """Get probability distribution of text features."""
        doc = self.nlp(text)
        features = []
        
        # Sentence lengths
        sent_lengths = [len(sent) for sent in doc.sents]
        features.extend(sent_lengths)
        
        # POS tag frequencies
        pos_counts = Counter(token.pos_ for token in doc)
        features.extend(pos_counts.values())
        
        # Normalize to get probability distribution
        features = np.array(features)
        return features / features.sum()

    def _perturbation_analysis(self, student_essay: str, ai_essays: List[str]) -> List[float]:
        """Analyze robustness to small perturbations."""
        def perturb_text(text):
            doc = self.nlp(text)
            perturbed = []
            for token in doc:
                if token.is_alpha and np.random.random() < 0.1:
                    perturbed.append(token.text.upper())
                else:
                    perturbed.append(token.text)
            return ' '.join(perturbed)
        
        n_perturbations = 5
        student_embeddings = []
        ai_embeddings = [[] for _ in ai_essays]
        
        # Get original embeddings
        student_orig = self.faiss_analyzer.get_embedding(student_essay)
        ai_origs = [self.faiss_analyzer.get_embedding(essay) for essay in ai_essays]
        
        # Get perturbed embeddings
        for _ in range(n_perturbations):
            student_pert = self.faiss_analyzer.get_embedding(perturb_text(student_essay))
            student_embeddings.append(student_pert)
            
            for i, essay in enumerate(ai_essays):
                ai_pert = self.faiss_analyzer.get_embedding(perturb_text(essay))
                ai_embeddings[i].append(ai_pert)
        
        # Calculate perturbation scores
        student_var = np.mean([
            cosine(student_orig.flatten(), pert.flatten())
            for pert in student_embeddings
        ])
        
        ai_vars = [
            np.mean([
                cosine(ai_orig.flatten(), pert.flatten())
                for pert in ai_perts
            ]) for ai_orig, ai_perts in zip(ai_origs, ai_embeddings)
        ]
        
        return [abs(student_var - ai_var) for ai_var in ai_vars]

    def _syntax_complexity(self, doc) -> float:
        """Calculate syntactic complexity score."""
        scores = []
        for sent in doc.sents:
            # Depth of dependency tree
            depth = max(len(list(token.ancestors)) for token in sent)
            # Number of dependency relations
            n_deps = len(list(sent.root.children))
            # Combine metrics
            scores.append(depth * n_deps)
        return np.mean(scores) if scores else 0

    def _ngram_analysis(self, student_essay: str, ai_essays: List[str], n: int = 3) -> List[float]:
        """Compare n-gram distributions."""
        def get_ngrams(text, n):
            tokens = word_tokenize(text.lower())
            return Counter(ngrams(tokens, n))
        
        student_ngrams = get_ngrams(student_essay, n)
        ai_ngrams = [get_ngrams(essay, n) for essay in ai_essays]
        
        # Calculate Jensen-Shannon divergence
        scores = []
        for ai_ng in ai_ngrams:
            all_ngrams = set(student_ngrams.keys()) | set(ai_ng.keys())
            p = np.array([student_ngrams.get(ng, 0) for ng in all_ngrams])
            q = np.array([ai_ng.get(ng, 0) for ng in all_ngrams])
            
            # Normalize
            p = p / p.sum() if p.sum() > 0 else p
            q = q / q.sum() if q.sum() > 0 else q
            
            m = 0.5 * (p + q)
            scores.append(0.5 * (entropy(p, m) + entropy(q, m)))
            
        return scores

    def _plot_results(self, student_essay: str, ai_essays: List[str], results: Dict[str, Any]):
        """Generate visualizations of the analysis results."""
        # 1. Similarity Matrix
        plt.figure(figsize=(10, 8))
        all_texts = [student_essay] + ai_essays
        sim_matrix = self.faiss_analyzer.batch_compare(all_texts)
        sns.heatmap(sim_matrix, 
                   xticklabels=['Student'] + [f'AI {i+1}' for i in range(len(ai_essays))],
                   yticklabels=['Student'] + [f'AI {i+1}' for i in range(len(ai_essays))],
                   cmap='YlOrRd')
        plt.title('Text Similarity Matrix')
        plt.show()

        # 2. Embedding Space Visualization
        embeddings = []
        for text in all_texts:
            emb = self.faiss_analyzer.get_embedding(text)
            embeddings.append(emb.flatten())
        embeddings = np.vstack(embeddings)
        
        pca = PCA(n_components=3)
        embeddings_3d = pca.fit_transform(embeddings)
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot student essay point
        ax.scatter(embeddings_3d[0, 0], 
                  embeddings_3d[0, 1], 
                  embeddings_3d[0, 2], 
                  c='r', 
                  marker='*', 
                  s=200, 
                  label='Student Essay')
        
        # Plot AI essays
        ax.scatter(embeddings_3d[1:, 0],
                  embeddings_3d[1:, 1],
                  embeddings_3d[1:, 2],
                  c='b',
                  marker='o',
                  s=100,
                  label='AI Essays')
        
        plt.title('Essay Embeddings in 3D Space')
        plt.legend()
        plt.show()

        # 3. Analysis Metrics Comparison
        metrics = {
            'Perplexity': (results['statistical_analysis']['perplexity'],
                          results['statistical_analysis']['avg_ai_perplexity']),
            'Semantic Coherence': (results['statistical_analysis']['semantic_coherence'],
                                 results['statistical_analysis']['avg_ai_coherence']),
            'Style Similarity': (1.0, 
                               results['statistical_analysis']['style_similarity']),
            'Syntax Complexity': (results['nlp_analysis']['syntax_complexity'],
                                results['nlp_analysis']['avg_ai_complexity'])
        }
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, [m[0] for m in metrics.values()], width, label='Student Essay')
        ax.bar(x + width/2, [m[1] for m in metrics.values()], width, label='Average AI')
        
        ax.set_xticks(x)
        ax.set_xticklabels(metrics.keys())
        ax.legend()
        plt.title('Analysis Metrics Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Example usage
    detector = EssayDetector()
    
    # Load texts
    with open('essay.txt', 'r') as f:
        student_essay = f.read()
        
    # TODO: Load AI generated essays
    ai_essays = []  # List of AI generated essays
    
    # Run analysis
    results = detector.analyze_essay(student_essay, ai_essays)
    
    # Print summary
    print("\nAnalysis Results:")
    print("-" * 50)
    print(f"FAISS Similarity Score: {np.mean(results['faiss_analysis']['combined_score']):.3f}")
    print(f"Perplexity Difference: {abs(results['statistical_analysis']['perplexity'] - results['statistical_analysis']['avg_ai_perplexity']):.3f}")
    print(f"Style Similarity: {results['statistical_analysis']['style_similarity']:.3f}")
    print(f"Average TV Distance: {np.mean(results['mathematical_analysis']['tv_distances']):.3f}")
    print(f"Perturbation Robustness: {np.mean(results['mathematical_analysis']['perturbation_scores']):.3f}")
    print(f"N-gram Similarity: {1 - np.mean(results['nlp_analysis']['ngram_scores']):.3f}") 