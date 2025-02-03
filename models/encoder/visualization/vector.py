import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import os

class VectorVisualizer:
    def __init__(self, vector_path):
        """
        Args:
            vector_path (str): .npy 파일 경로
        """
        # 벡터 데이터 로드
        self.vectors = np.load(vector_path, allow_pickle=True).item()
        
        # 이미지 경로와 벡터를 분리
        self.image_paths = list(self.vectors.keys())
        vectors_list = list(self.vectors.values())
        
        # 벡터들을 스택하기 전에 형태 확인 및 처리
        processed_vectors = []
        for vec in vectors_list:
            # 벡터가 스칼라인 경우 2차원으로 확장
            if vec.shape == ():
                processed_vectors.append([vec.item(), 0])  # 두 번째 차원 추가
            else:
                processed_vectors.append(vec)
        
        # 벡터들을 스택
        self.feature_vectors = np.vstack(processed_vectors)
        
        print(f"Sample vector shape after processing: {self.feature_vectors[0].shape}")
        print(f"Loaded {len(self.image_paths)} vectors with shape {self.feature_vectors.shape}")
    
    def visualize_pca(self, save_path=None):
        """PCA로 2D 시각화"""
        # 데이터 차원 확인
        n_components = min(2, self.feature_vectors.shape[1])
        
        # PCA 수행
        pca = PCA(n_components=n_components)
        vectors_2d = pca.fit_transform(self.feature_vectors)
        
        # 1차원인 경우 2차원으로 확장
        if vectors_2d.shape[1] == 1:
            vectors_2d = np.hstack([vectors_2d, np.zeros_like(vectors_2d)])
        
        # 시각화
        plt.figure(figsize=(12, 8))
        plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.5)
        plt.title('PCA Visualization of Feature Vectors')
        if n_components == 2:
            plt.xlabel(f'First PC (explained variance: {pca.explained_variance_ratio_[0]:.3f})')
            plt.ylabel(f'Second PC (explained variance: {pca.explained_variance_ratio_[1]:.3f})')
        else:
            plt.xlabel('First PC')
            plt.ylabel('Added dimension')
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved PCA visualization to {save_path}")
        plt.close()
        
        return vectors_2d
    
    def visualize_tsne(self, save_path=None, perplexity=30):
        """t-SNE로 2D 시각화"""
        # t-SNE 수행
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        vectors_2d = tsne.fit_transform(self.feature_vectors)
        
        # 시각화
        plt.figure(figsize=(12, 8))
        plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.5)
        plt.title('t-SNE Visualization of Feature Vectors')
        plt.xlabel('First dimension')
        plt.ylabel('Second dimension')
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved t-SNE visualization to {save_path}")
        plt.close()
        
        return vectors_2d
    
    def visualize_distance_distribution(self, save_path=None):
        """벡터 간 거리 분포 시각화"""
        # 모든 벡터 쌍 간의 코사인 유사도 계산
        similarities = []
        for i in range(len(self.feature_vectors)):
            for j in range(i + 1, len(self.feature_vectors)):
                similarity = np.dot(self.feature_vectors[i], self.feature_vectors[j]) / (
                    np.linalg.norm(self.feature_vectors[i]) * np.linalg.norm(self.feature_vectors[j])
                )
                similarities.append(similarity)
        
        # 분포 시각화"
        plt.figure(figsize=(10, 6))
        sns.histplot(similarities, bins=50)
        plt.title('Distribution of Cosine Similarities between Vectors')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Count')
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved similarity distribution to {save_path}")
        plt.close()
        
        return similarities

if __name__ == "__main__":
    # 설정
    vector_path = "/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/vectors/origin.npy"
    output_dir = "/home/minelab/desktop/ANN/jojun/himeow-eye/datasets/vectors/visualization"
    os.makedirs(output_dir, exist_ok=True)
    
    # 시각화 객체 생성
    visualizer = VectorVisualizer(vector_path)
    
    # PCA 시각화
    pca_vectors = visualizer.visualize_pca(
        save_path=os.path.join(output_dir, 'pca_visualization.png')
    )
    
    # t-SNE 시각화
    tsne_vectors = visualizer.visualize_tsne(
        save_path=os.path.join(output_dir, 'tsne_visualization.png')
    )
    
    # 거리 분포 시각화
    similarities = visualizer.visualize_distance_distribution(
        save_path=os.path.join(output_dir, 'similarity_distribution.png')
    )