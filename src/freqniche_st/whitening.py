"""
Train-Only PCA Whitening and Identity Projection for FreqNiche-ST.
Strictly enforces:
- Fit ONLY on train split cells
- Target dimension k_target = 32, identity dimension k_identity = 32
- Clamping of eigenvalues: lambda_i = max(lambda_i, 1e-4) to prevent division by zero / explosive noise
- Dynamic n_components_fitted_ for dimension-safe reshaping
- Orthogonal unit variance in latent space for stable diffusion
"""

from typing import Optional
import numpy as np
import scipy.sparse as sp


class TrainOnlyWhitening:
    """
    Train-only PCA whitening transform for high-frequency residual supervision targets.
    Projects high-dimensional sparse residual H -> whitened latent H_0 in R^k.
    """
    def __init__(self, k_components: int = 32, lambda_min: float = 1e-4):
        self.k_components = k_components
        self.lambda_min = lambda_min
        self.mean_ = None
        self.components_ = None        # V_k: (F, k)
        self.singular_values_ = None   # sqrt(lambda_i): (k,)
        self.whitening_matrix_ = None  # V_k * Lambda^{-1/2}: (F, k)
        self.n_components_fitted_ = None
        self.is_fitted = False

    def fit(self, H_train: np.ndarray) -> "TrainOnlyWhitening":
        """
        Fits PCA whitening basis on training residuals.

        Args:
            H_train: (N_train, F) numpy array or sparse matrix of high-frequency residuals.
        """
        if sp.issparse(H_train):
            H_train = H_train.toarray()
        H_train = np.asarray(H_train, dtype=np.float32)

        n_samples, n_features = H_train.shape
        if n_samples < 2:
            raise ValueError(f"TrainOnlyWhitening requires at least 2 samples to fit, got {n_samples}")

        self.mean_ = np.mean(H_train, axis=0, keepdims=True)  # (1, F)

        H_centered = H_train - self.mean_

        # SVD for numerical stability: H_centered = U * S * V^T
        k = min(self.k_components, n_features, n_samples - 1)
        self.n_components_fitted_ = k
        
        # Economy SVD
        U, S, Vt = np.linalg.svd(H_centered, full_matrices=False)
        
        V_k = Vt[:k].T  # (F, k)
        eigenvalues = (S[:k] ** 2) / max(n_samples - 1, 1)

        # Clamping variance lower bound to lambda_min
        eigenvalues_clamped = np.maximum(eigenvalues, self.lambda_min)
        std_inv = 1.0 / np.sqrt(eigenvalues_clamped)  # (k,)

        self.components_ = V_k.astype(np.float32)
        self.singular_values_ = np.sqrt(eigenvalues_clamped).astype(np.float32)
        self.whitening_matrix_ = (V_k * std_inv).astype(np.float32)  # (F, k)
        self.is_fitted = True

        return self

    def transform(self, H: np.ndarray) -> np.ndarray:
        """
        Transforms residuals H -> whitened latent H_0 in R^k.

        Args:
            H: (B, F) or (B, N, F) residuals array.
        """
        if not self.is_fitted:
            raise RuntimeError("TrainOnlyWhitening must be fitted before transform.")
        
        if sp.issparse(H):
            H = H.toarray()
        
        H = np.asarray(H, dtype=np.float32)
        shape_orig = H.shape
        if len(shape_orig) == 3:
            B, N, F = shape_orig
            H_flat = H.reshape(-1, F)
            H_white = np.matmul(H_flat - self.mean_, self.whitening_matrix_)
            return H_white.reshape(B, N, self.n_components_fitted_)
        else:
            return np.matmul(H - self.mean_, self.whitening_matrix_)

    def inverse_transform(self, H_0: np.ndarray) -> np.ndarray:
        """
        Inverts whitened latent H_0 in R^k back to residual feature space R^F.
        """
        if not self.is_fitted:
            raise RuntimeError("TrainOnlyWhitening must be fitted before inverse_transform.")
        
        H_0 = np.asarray(H_0, dtype=np.float32)
        unwhitened = H_0 * self.singular_values_
        return np.matmul(unwhitened, self.components_.T) + self.mean_


class TrainOnlyPCA:
    """
    Train-only standard PCA projection for cell identity embeddings.
    Projects raw expression X -> z^identity in R^k without whitening.
    """
    def __init__(self, k_components: int = 32):
        self.k_components = k_components
        self.mean_ = None
        self.components_ = None  # (F, k)
        self.n_components_fitted_ = None
        self.is_fitted = False

    def fit(self, X_train: np.ndarray) -> "TrainOnlyPCA":
        if sp.issparse(X_train):
            X_train = X_train.toarray()
        X_train = np.asarray(X_train, dtype=np.float32)

        n_samples, n_features = X_train.shape
        if n_samples < 2:
            raise ValueError(f"TrainOnlyPCA requires at least 2 samples to fit, got {n_samples}")

        self.mean_ = np.mean(X_train, axis=0, keepdims=True)

        X_centered = X_train - self.mean_
        k = min(self.k_components, n_features, n_samples - 1)
        self.n_components_fitted_ = k
        
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.components_ = Vt[:k].T.astype(np.float32)
        self.is_fitted = True

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("TrainOnlyPCA must be fitted before transform.")
        
        if sp.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        return np.matmul(X - self.mean_, self.components_)


class TruncatedTrainOnlyWhitening:
    """
    Scalable Train-Only Truncated PCA Whitening for ultra-high-dimensional datasets
    (e.g., 5,001 genes x 60k+ cells).
    
    Avoids full dense SVD of large (N, F) matrices.
    Uses randomized SVD or chunked covariance eigendecomposition to extract exactly top k components in O(N * F * k).
    
    Guarantees:
    - Whitening projection matrix shape: strictly (F, k), e.g. (5001, 32)
    - Orthogonal basis V_k: (F, k)
    - Clamped eigenvalues: max(lambda_i, lambda_min) to prevent division by zero
    - Numerical finiteness: all values finite, no NaN, no Inf
    - Memory footprint bounded
    """
    def __init__(self, k_components: int = 32, lambda_min: float = 1e-4, random_state: int = 42):
        self.k_components = k_components
        self.lambda_min = lambda_min
        self.random_state = random_state
        self.mean_ = None
        self.components_ = None        # V_k: (F, k)
        self.singular_values_ = None   # sqrt(lambda_i): (k,)
        self.whitening_matrix_ = None  # V_k * Lambda^{-1/2}: (F, k)
        self.n_components_fitted_ = None
        self.is_fitted = False

    def fit(self, H_train: np.ndarray) -> "TruncatedTrainOnlyWhitening":
        """
        Fits truncated PCA whitening basis on training residuals using randomized SVD.
        """
        from sklearn.utils.extmath import randomized_svd

        if sp.issparse(H_train):
            H_train = H_train.toarray()
        H_train = np.asarray(H_train, dtype=np.float32)

        n_samples, n_features = H_train.shape
        if n_samples < 2:
            raise ValueError(f"TruncatedTrainOnlyWhitening requires at least 2 samples, got {n_samples}")

        self.mean_ = np.mean(H_train, axis=0, keepdims=True)  # (1, F)
        H_centered = H_train - self.mean_

        k = min(self.k_components, n_features, n_samples - 1)
        self.n_components_fitted_ = k

        # Randomized SVD: H_centered ~= U * S * Vt, shape of Vt: (k, F)
        U, S, Vt = randomized_svd(
            H_centered,
            n_components=k,
            n_iter=4,
            random_state=self.random_state,
        )

        V_k = Vt.T  # (F, k)
        eigenvalues = (S ** 2) / max(n_samples - 1, 1)

        # Variance clamping to lambda_min
        eigenvalues_clamped = np.maximum(eigenvalues, self.lambda_min)
        std_inv = 1.0 / np.sqrt(eigenvalues_clamped)  # (k,)

        self.components_ = V_k.astype(np.float32)
        self.singular_values_ = np.sqrt(eigenvalues_clamped).astype(np.float32)
        self.whitening_matrix_ = (V_k * std_inv).astype(np.float32)  # (F, k)
        self.is_fitted = True

        return self

    def fit_from_covariance(self, cov_matrix: np.ndarray, mean_vector: np.ndarray, n_samples: int) -> "TruncatedTrainOnlyWhitening":
        """
        Fits truncated PCA whitening basis directly from pre-accumulated sample covariance matrix C in R^{F x F}.
        Enables multi-sample out-of-core streaming without loading full dataset into memory.
        """
        import scipy.linalg as la

        cov_matrix = np.asarray(cov_matrix, dtype=np.float32)
        n_features = cov_matrix.shape[0]
        assert cov_matrix.shape == (n_features, n_features), "Covariance matrix must be square"

        self.mean_ = np.asarray(mean_vector, dtype=np.float32).reshape(1, n_features)
        k = min(self.k_components, n_features, n_samples - 1)
        self.n_components_fitted_ = k

        # Symmetrize covariance for numerical stability
        cov_sym = 0.5 * (cov_matrix + cov_matrix.T)

        # Compute top k eigenvalues and eigenvectors of symmetric matrix
        # subset_by_index returns eigenvalues in ascending order
        eigvals, eigvecs = la.eigh(cov_sym, subset_by_index=[n_features - k, n_features - 1])

        # Reverse to descending order
        eigvals = eigvals[::-1]
        V_k = eigvecs[:, ::-1]

        eigenvalues_clamped = np.maximum(eigvals, self.lambda_min)
        std_inv = 1.0 / np.sqrt(eigenvalues_clamped)

        self.components_ = V_k.astype(np.float32)
        self.singular_values_ = np.sqrt(eigenvalues_clamped).astype(np.float32)
        self.whitening_matrix_ = (V_k * std_inv).astype(np.float32)  # (F, k)
        self.is_fitted = True

        return self

    def transform(self, H: np.ndarray) -> np.ndarray:
        """
        Transforms residuals H -> whitened latent H_0 in R^k.
        """
        if not self.is_fitted:
            raise RuntimeError("TruncatedTrainOnlyWhitening must be fitted before transform.")

        if sp.issparse(H):
            H = H.toarray()

        H = np.asarray(H, dtype=np.float32)
        shape_orig = H.shape
        if len(shape_orig) == 3:
            B, N, F = shape_orig
            H_flat = H.reshape(-1, F)
            H_white = np.matmul(H_flat - self.mean_, self.whitening_matrix_)
            return H_white.reshape(B, N, self.n_components_fitted_)
        else:
            return np.matmul(H - self.mean_, self.whitening_matrix_)

    def inverse_transform(self, H_0: np.ndarray) -> np.ndarray:
        """
        Inverts whitened latent H_0 in R^k back to residual feature space R^F.
        """
        if not self.is_fitted:
            raise RuntimeError("TruncatedTrainOnlyWhitening must be fitted before inverse_transform.")

        H_0 = np.asarray(H_0, dtype=np.float32)
        unwhitened = H_0 * self.singular_values_
        return np.matmul(unwhitened, self.components_.T) + self.mean_

