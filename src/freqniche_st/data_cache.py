"""In-memory, run-scoped caches for immutable inputs, never masked conditions.

No random numbers are consumed here. X, P and A must not be mutated after
construction; create a new cache after changing expression/graph/split.
"""

from collections import OrderedDict
import numpy as np

from .whitening import TrainOnlyPCA, TrainOnlyWhitening


class ResidualCache:
    """Lazily memoize true X - P@X rows, with the original float32 arithmetic."""

    def __init__(self, X, P):
        self.X, self.P = X, P
        self.values = np.empty(X.shape, dtype=np.float32)
        self.ready = np.zeros(X.shape[0], dtype=bool)
        self.computed_rows = 0

    def rows(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        missing = np.unique(indices[~self.ready[indices]])
        if missing.size:
            # Only these target rows, on the TRUE global graph. No condition
            # features or whitening statistics are derived from held-out data.
            self.values[missing] = (
                self.X[missing].astype(np.float32)
                - (self.P[missing, :] @ self.X).astype(np.float32)
            )
            self.ready[missing] = True
            self.computed_rows += len(missing)
        return self.values[indices]


class GraphDataCache:
    """Bounded LRU of sorted neighborhoods and their unnormalized local A.

    Masking and row normalization still happen AFTER selecting visible nodes.
    Caching PX on the complete graph as an encoder feature would leak targets;
    this cache deliberately does not do that.
    """

    def __init__(self, A, X, P, residuals=None, max_entries=100000):
        self.A, self.X, self.P = A, X, P
        self.residuals = residuals if residuals is not None else ResidualCache(X, P)
        if self.residuals.X is not X or self.residuals.P is not P:
            raise ValueError("Residual cache belongs to different true inputs")
        self.max_entries = max_entries
        self.entries = OrderedDict()
        self.hits = self.misses = 0

    def validate(self, A, X, P):
        if A is not self.A or X is not self.X or P is not self.P:
            raise ValueError("Data cache cannot be reused with changed graph/expression")

    def entry(self, center_idx, k_max):
        key = (int(center_idx), int(k_max))
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key]
        self.misses += 1
        # Import locally to keep the uncached sampler an independent oracle.
        from .subgraph import _get_sorted_subgraph_nodes
        nodes, others = _get_sorted_subgraph_nodes(center_idx, self.A, k_max)
        local_A = self.A[nodes, :][:, nodes].toarray().astype(np.float32)
        sorter = np.argsort(nodes)
        entry = (nodes, others, local_A, sorter)
        self.entries[key] = entry
        if len(self.entries) > self.max_entries:
            self.entries.popitem(last=False)
        return entry

    def nodes(self, center_idx, k_max):
        entry = self.entry(center_idx, k_max)
        return entry[0], entry[1]

    def local_adjacency(self, query_idx, visible_nodes, k_max):
        nodes, _, local_A, sorter = self.entry(query_idx, k_max)
        positions = sorter[np.searchsorted(nodes[sorter], visible_nodes)]
        if not np.array_equal(nodes[positions], visible_nodes):
            raise ValueError("Visible nodes are outside the cached neighborhood")
        # Same order and values as A[visible,:][:,visible], without global CSR
        # slicing for every center on every epoch.
        return local_A[np.ix_(positions, positions)]

    def stats(self):
        return {
            "neighborhoods": len(self.entries), "hits": self.hits,
            "misses": self.misses, "residual_rows": self.residuals.computed_rows,
            "local_array_bytes": sum(sum(a.nbytes for a in e) for e in self.entries.values()),
            "residual_allocated_bytes": self.residuals.values.nbytes + self.residuals.ready.nbytes,
        }


class SharedPreprocessing:
    """Reuse identical train-only PCA fits and true-graph cache across runs."""

    def __init__(self, X, P, A, train_indices):
        self.X, self.P, self.A = X, P, A
        self.train_indices = np.asarray(train_indices, dtype=np.int64).copy()
        self.residuals = ResidualCache(X, P)
        # Preserve the original PCA fit ordering and numerical implementation.
        X_train = X[self.train_indices]
        H_train = X_train - (P[self.train_indices] @ X).astype(np.float32)
        self.whitener = TrainOnlyWhitening(32, 1e-4).fit(H_train)
        self.pca_id = TrainOnlyPCA(32).fit(X_train)
        self.residuals.values[self.train_indices] = H_train
        self.residuals.ready[self.train_indices] = True
        self.residuals.computed_rows = len(np.unique(self.train_indices))
        self.true_graph_cache = GraphDataCache(A, X, P, self.residuals)

    def validate(self, X, P, A, train_indices):
        if (X is not self.X or P is not self.P or A is not self.A
                or not np.array_equal(train_indices, self.train_indices)):
            raise ValueError("Preprocessing cache does not match current data/split")

    def for_graph(self, A):
        if A is self.A:
            return self.true_graph_cache
        # A shuffled condition graph MUST get a separate neighborhood cache.
        return GraphDataCache(A, self.X, self.P, self.residuals)
