import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple

from ..core.memory_unit import MemoryUnit


def find_clusters_with_dbscan(units_with_embeddings: List[MemoryUnit], 
                             eps: float = 0.3,
                             min_samples: int = 3,
                             metric: str = 'cosine',
                             normalize_embeddings: bool = True,
                             use_metadata_features: bool = False) -> Dict[int, List[str]]:
    """Find clusters with dbscan."""
    if not units_with_embeddings:
        logging.warning("No units with embeddings were provided")
        return {}
    
    logging.info(f"Starting DBSCAN clustering for {len(units_with_embeddings)} units")
    
    feature_matrix, uid_mapping = _prepare_feature_matrix(
        units_with_embeddings, 
        normalize_embeddings, 
        use_metadata_features
    )
    
    if feature_matrix is None or len(uid_mapping) == 0:
        logging.error("Feature matrix preparation failed or returned an empty matrix")
        return {}
    
    if np.any(np.isnan(feature_matrix)) or np.any(np.isinf(feature_matrix)):
        logging.warning("Feature matrix contains NaN or infinite values; attempting cleanup")
        feature_matrix = _clean_feature_matrix(feature_matrix)
        if feature_matrix is None:
            logging.error("Data remains invalid after cleanup")
            return {}
    
    try:
        from sklearn.cluster import DBSCAN
    except ImportError as e:
        logging.error(f"DBSCAN clustering requires scikit-learn: {e}")
        return {}

    try:
        if metric == 'cosine':
            distance_matrix = _compute_safe_cosine_distance(feature_matrix)
            if distance_matrix is None:
                logging.error("Cosine distance matrix computation failed; falling back to Euclidean distance")
                metric = 'euclidean'
                
        if metric == 'cosine' and distance_matrix is not None:
            dbscan = DBSCAN(
                eps=eps, 
                min_samples=min_samples, 
                metric='precomputed'
            )
            labels = dbscan.fit_predict(distance_matrix)
        else:
            dbscan = DBSCAN(
                eps=eps, 
                min_samples=min_samples, 
                metric=metric
            )
            labels = dbscan.fit_predict(feature_matrix)
            
        logging.info(f"DBSCAN clustering completed with {len(set(labels))} clusters including noise")
        
    except Exception as e:
        logging.error(f"DBSCAN clustering failed: {e}")
        try:
            logging.info("Retrying clustering with fallback parameters")
            dbscan_fallback = DBSCAN(
                eps=min(eps * 2, 1.0), 
                min_samples=max(min_samples - 1, 2), 
                metric='euclidean'
            )
            labels = dbscan_fallback.fit_predict(feature_matrix)
            logging.info(f"Fallback clustering succeeded with {len(set(labels))} clusters")
        except Exception as e2:
            logging.error(f"Fallback clustering also failed: {e2}")
            return {}
    
    clusters = {}
    noise_count = 0
    
    for i, label in enumerate(labels):
        if i >= len(uid_mapping):
            logging.warning(f"Label index {i} is outside the UID mapping range")
            continue
            
        uid = uid_mapping[i]
        
        if label == -1:
            noise_count += 1
            if -1 not in clusters:
                clusters[-1] = []
            clusters[-1].append(uid)
        else:
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(uid)
    
    valid_clusters = {k: v for k, v in clusters.items() if k != -1}
    logging.info(f"Valid clusters: {len(valid_clusters)}")
    logging.info(f"Noise points: {noise_count}")
    
    for cluster_id, uids in valid_clusters.items():
        logging.info(f"Cluster {cluster_id}: {len(uids)} units")
    
    return clusters


def _compute_safe_cosine_distance(feature_matrix: np.ndarray) -> Optional[np.ndarray]:
    """Compute safe cosine distance."""
    try:
        from sklearn.metrics.pairwise import cosine_similarity

        
        norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True)
        
        
        zero_mask = norms.flatten() == 0
        if np.any(zero_mask):
            logging.warning(f"Found {np.sum(zero_mask)} zero vectors; replacing them with small random values")
            feature_matrix[zero_mask] = np.random.normal(0, 1e-8, (np.sum(zero_mask), feature_matrix.shape[1]))
            norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True)
        
        normalized_features = feature_matrix / norms
        
        cosine_sim = cosine_similarity(normalized_features)
        
        distance_matrix = 1 - cosine_sim
        
        np.fill_diagonal(distance_matrix, 0)
        distance_matrix = np.maximum(distance_matrix, 0)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        
        if np.any(np.isnan(distance_matrix)) or np.any(np.isinf(distance_matrix)):
            logging.error("Distance matrix contains NaN or infinite values")
            return None
            
        if np.any(distance_matrix < 0):
            logging.error("Distance matrix contains negative values")
            return None
        
        logging.info(f"Cosine distance matrix computed successfully, shape={distance_matrix.shape}")
        logging.debug(f"Distance matrix stats: min={np.min(distance_matrix):.6f}, max={np.max(distance_matrix):.6f}")
        
        return distance_matrix
        
    except Exception as e:
        logging.error(f"Failed to compute cosine distance matrix: {e}")
        return None


def _clean_feature_matrix(feature_matrix: np.ndarray) -> Optional[np.ndarray]:
    """Run clean feature matrix."""
    try:
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if np.any(np.isnan(feature_matrix)) or np.any(np.isinf(feature_matrix)):
            logging.error("Invalid values remain after cleanup")
            return None
        
        
        row_norms = np.linalg.norm(feature_matrix, axis=1)
        zero_rows = row_norms == 0
        
        if np.all(zero_rows):
            logging.error("All feature vectors are zero vectors")
            return None
        
        if np.any(zero_rows):
            logging.warning(f"Found {np.sum(zero_rows)} zero-vector rows")
            
            feature_matrix[zero_rows] = np.random.normal(0, 1e-8, (np.sum(zero_rows), feature_matrix.shape[1]))
        
        return feature_matrix
        
    except Exception as e:
        logging.error(f"Failed to clean feature matrix: {e}")
        return None


def _prepare_feature_matrix(units_with_embeddings: List[MemoryUnit], 
                           normalize_embeddings: bool,
                           use_metadata_features: bool) -> Tuple[Optional[np.ndarray], List[str]]:
    """Run prepare feature matrix."""
    
    embeddings = []
    uid_mapping = []
    metadata_features = []
    
    for unit in units_with_embeddings:
        if unit.embedding is None:
            continue
        
        
        embedding = np.array(unit.embedding)
        if len(embedding.shape) != 1:
            logging.warning(f"Unit {unit.uid} has an unexpected embedding shape: {embedding.shape}")
            continue
            
        if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
            logging.warning(f"Unit {unit.uid} embedding contains NaN or infinite values")
            continue
            
        embeddings.append(embedding)
        uid_mapping.append(unit.uid)
        
        if use_metadata_features:
            meta_features = _extract_metadata_features(unit)
            metadata_features.append(meta_features)
    
    if not embeddings:
        logging.error("No valid embeddings were found")
        return None, []
    
    
    embedding_dims = [len(emb) for emb in embeddings]
    if len(set(embedding_dims)) > 1:
        logging.warning(f"Embedding dimensions are inconsistent: {set(embedding_dims)}")
        min_dim = min(embedding_dims)
        embeddings = [emb[:min_dim] for emb in embeddings]
        logging.info(f"Truncated all embeddings to {min_dim} dimensions")
    
    try:
        feature_matrix = np.array(embeddings, dtype=np.float32)
    except Exception as e:
        logging.error(f"Failed to convert embeddings to a NumPy array: {e}")
        return None, []
    
    if feature_matrix.size == 0:
        logging.error("Feature matrix is empty")
        return None, []
    
    
    if normalize_embeddings:
        try:
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            feature_matrix = scaler.fit_transform(feature_matrix)
        except Exception as e:
            logging.warning(f"Normalization failed: {e}")
            norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1
            feature_matrix = feature_matrix / norms
    
    if use_metadata_features and metadata_features:
        try:
            from sklearn.preprocessing import StandardScaler

            metadata_matrix = np.array(metadata_features, dtype=np.float32)
            meta_scaler = StandardScaler()
            metadata_matrix = meta_scaler.fit_transform(metadata_matrix)
            feature_matrix = np.concatenate([feature_matrix, metadata_matrix], axis=1)
            logging.info(f"Feature matrix shape with metadata: {feature_matrix.shape}")
        except Exception as e:
            logging.warning(f"Failed to merge metadata features: {e}")
            logging.info(f"Using original feature matrix shape: {feature_matrix.shape}")
    else:
        logging.info(f"Feature matrix shape: {feature_matrix.shape}")
    
    return feature_matrix, uid_mapping


def _extract_metadata_features(unit: MemoryUnit) -> List[float]:
    """Extract metadata features."""
    features = []
    
    if unit.metadata:
        if 'timestamp' in unit.metadata:
            try:
                from datetime import datetime
                timestamp = unit.metadata['timestamp']
                if isinstance(timestamp, str):
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    features.extend([
                        dt.hour / 24.0,
                        dt.weekday() / 6.0,
                        dt.month / 12.0
                    ])
                else:
                    features.extend([0.0, 0.0, 0.0])
            except:
                features.extend([0.0, 0.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        content_type = unit.metadata.get('content_type', 'unknown')
        type_mapping = {
            'conversation': [1.0, 0.0, 0.0, 0.0],
            'document': [0.0, 1.0, 0.0, 0.0],
            'task': [0.0, 0.0, 1.0, 0.0],
            'unknown': [0.0, 0.0, 0.0, 1.0]
        }
        features.extend(type_mapping.get(content_type, [0.0, 0.0, 0.0, 1.0]))
        
        importance = unit.metadata.get('importance', 0.5)
        if isinstance(importance, (int, float)):
            features.append(float(importance))
        else:
            features.append(0.5)
    else:
        features.extend([0.0] * 8)
    
    return features


def optimize_dbscan_parameters(units_with_embeddings: List[MemoryUnit],
                              eps_range: Tuple[float, float] = (0.1, 0.8),
                              min_samples_range: Tuple[int, int] = (2, 10),
                              metric: str = 'cosine',
                              n_trials: int = 20) -> Dict[str, Any]:
    """Run optimize dbscan parameters."""
    if len(units_with_embeddings) < 10:
        logging.warning("Too few samples; using default parameters")
        return {
            'best_params': {'eps': 0.3, 'min_samples': 3},
            'best_score': 0.0,
            'evaluation': 'insufficient_data'
        }
    
    logging.info(f"Starting DBSCAN parameter tuning for {len(units_with_embeddings)} samples")
    
    feature_matrix, uid_mapping = _prepare_feature_matrix(
        units_with_embeddings, 
        normalize_embeddings=True, 
        use_metadata_features=False
    )
    
    if feature_matrix is None:
        return {
            'best_params': {'eps': 0.3, 'min_samples': 3},
            'best_score': 0.0,
            'evaluation': 'feature_preparation_failed'
        }
    
    best_params = None
    best_score = -1
    evaluation_results = []
    
    try:
        from sklearn.cluster import DBSCAN
    except ImportError as e:
        logging.error(f"DBSCAN parameter tuning requires scikit-learn: {e}")
        return {
            'best_params': {'eps': 0.3, 'min_samples': 3},
            'best_score': 0.0,
            'evaluation': 'sklearn_unavailable'
        }

    eps_values = np.linspace(eps_range[0], eps_range[1], max(n_trials // 4, 3))
    min_samples_values = range(min_samples_range[0], min(min_samples_range[1] + 1, len(units_with_embeddings) // 2))
    
    distance_matrix = None
    if metric == 'cosine':
        distance_matrix = _compute_safe_cosine_distance(feature_matrix)
        if distance_matrix is None:
            logging.warning("Cosine distance computation failed; using Euclidean distance")
            metric = 'euclidean'
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            try:
                if metric == 'cosine' and distance_matrix is not None:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
                    labels = dbscan.fit_predict(distance_matrix)
                else:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
                    labels = dbscan.fit_predict(feature_matrix)
                
                score = _evaluate_clustering_quality(labels, feature_matrix)
                
                evaluation_results.append({
                    'eps': float(eps),
                    'min_samples': int(min_samples),
                    'score': float(score),
                    'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
                    'n_noise': int(list(labels).count(-1))
                })
                
                if score > best_score:
                    best_score = score
                    best_params = {'eps': float(eps), 'min_samples': int(min_samples)}
                    
            except Exception as e:
                logging.debug(f"Parameters eps={eps:.3f}, min_samples={min_samples} failed: {e}")
                continue
    
    logging.info(f"Parameter tuning completed, best score={best_score:.3f}")
    logging.info(f"Best parameters: {best_params}")
    
    return {
        'best_params': best_params or {'eps': 0.3, 'min_samples': 3},
        'best_score': float(best_score),
        'evaluation_results': evaluation_results,
        'evaluation': 'completed'
    }


def _evaluate_clustering_quality(labels: np.ndarray, features: np.ndarray) -> float:
    """Run evaluate clustering quality."""
    try:
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
    except ImportError:
        logging.warning("sklearn.metrics is unavailable; returning a simple evaluation")
        return _simple_clustering_evaluation(labels)
    
    unique_labels = set(labels)
    if len(unique_labels) <= 1 or (len(unique_labels) == 2 and -1 in unique_labels):
        return 0.0
    
    try:
        mask = labels != -1
        if np.sum(mask) < 2:
            return 0.0
        
        filtered_labels = labels[mask]
        filtered_features = features[mask]
        
        if len(set(filtered_labels)) < 2:
            return 0.0
        
        silhouette = silhouette_score(filtered_features, filtered_labels)
        
        ch_score = calinski_harabasz_score(filtered_features, filtered_labels)
        ch_normalized = min(ch_score / 1000.0, 1.0)
        
        noise_ratio = list(labels).count(-1) / len(labels)
        noise_penalty = 1.0 - noise_ratio
        
        final_score = (silhouette * 0.5 + ch_normalized * 0.3 + noise_penalty * 0.2)
        
        return max(0.0, final_score)
        
    except Exception as e:
        logging.debug(f"Cluster-quality evaluation failed: {e}")
        return _simple_clustering_evaluation(labels)


def _simple_clustering_evaluation(labels: np.ndarray) -> float:
    """Run simple clustering evaluation."""
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(labels).count(-1)
    
    if n_clusters == 0:
        return 0.0
    
    cluster_score = min(n_clusters / 10.0, 1.0)
    noise_penalty = 1.0 - (n_noise / len(labels))
    
    return (cluster_score * 0.6 + noise_penalty * 0.4)


def analyze_cluster_characteristics(clusters: Dict[int, List[str]], 
                                  units_with_embeddings: List[MemoryUnit]) -> Dict[str, Any]:
    """Run analyze cluster characteristics."""
    uid_to_unit = {unit.uid: unit for unit in units_with_embeddings}
    
    analysis = {
        'total_clusters': len([k for k in clusters.keys() if k != -1]),
        'noise_points': len(clusters.get(-1, [])),
        'cluster_details': {}
    }
    
    for cluster_id, uids in clusters.items():
        if cluster_id == -1:
            continue
            
        cluster_units = [uid_to_unit[uid] for uid in uids if uid in uid_to_unit]
        
        content_types = {}
        for unit in cluster_units:
            content_type = unit.metadata.get('content_type', 'unknown') if unit.metadata else 'unknown'
            content_types[content_type] = content_types.get(content_type, 0) + 1
        
        timestamps = []
        for unit in cluster_units:
            if unit.metadata and 'timestamp' in unit.metadata:
                timestamps.append(unit.metadata['timestamp'])
        
        all_text = ""
        for unit in cluster_units:
            text_content = unit.raw_data.get('text_content', '') if unit.raw_data else ''
            all_text += text_content + " "
        
        keywords = _extract_simple_keywords(all_text)
        
        analysis['cluster_details'][cluster_id] = {
            'size': len(uids),
            'content_type_distribution': content_types,
            'timestamp_range': _analyze_timestamp_range(timestamps),
            'top_keywords': keywords[:10],
            'representative_units': uids[:3]
        }
    
    return analysis


def _extract_simple_keywords(text: str, top_k: int = 20) -> List[str]:
    """Extract simple keywords."""
    import re
    from collections import Counter
    
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text.lower())
    
    words = text.split()
    
    stop_words = {'的', '是', '在', '有', '和', '了', '我', '你', '他', 'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by'}
    
    filtered_words = [word for word in words if len(word) > 2 and word not in stop_words]
    
    word_counts = Counter(filtered_words)
    
    return [word for word, count in word_counts.most_common(top_k)]


def _analyze_timestamp_range(timestamps: List[str]) -> Dict[str, Any]:
    """Run analyze timestamp range."""
    if not timestamps:
        return {'range': 'no_timestamps', 'span_days': 0}
    
    try:
        from datetime import datetime
        
        parsed_times = []
        for ts in timestamps:
            try:
                if isinstance(ts, str):
                    dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    parsed_times.append(dt)
            except:
                continue
        
        if not parsed_times:
            return {'range': 'invalid_timestamps', 'span_days': 0}
        
        min_time = min(parsed_times)
        max_time = max(parsed_times)
        span = (max_time - min_time).days
        
        return {
            'range': f"{min_time.strftime('%Y-%m-%d')} to {max_time.strftime('%Y-%m-%d')}",
            'span_days': span,
            'earliest': min_time.isoformat(),
            'latest': max_time.isoformat()
        }
        
    except Exception as e:
        logging.warning(f"Timestamp analysis failed: {e}")
        return {'range': 'analysis_failed', 'span_days': 0}
