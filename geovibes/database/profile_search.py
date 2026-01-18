"""Profile search performance to understand caching behavior."""

import time
from dataclasses import dataclass, field
from typing import Optional

import faiss
import numpy as np


@dataclass
class SearchProfile:
    """Timing breakdown for a single search."""

    faiss_search_ms: float = 0.0
    duckdb_query_ms: float = 0.0
    total_ms: float = 0.0
    n_results: int = 0
    n_row_groups_hint: Optional[int] = None

    def __str__(self):
        return (
            f"Search Profile:\n"
            f"  FAISS search:  {self.faiss_search_ms:>8.1f} ms\n"
            f"  DuckDB query:  {self.duckdb_query_ms:>8.1f} ms\n"
            f"  Total:         {self.total_ms:>8.1f} ms\n"
            f"  Results:       {self.n_results}"
        )


@dataclass
class SearchProfiler:
    """Profile search operations to understand caching behavior."""

    profiles: list = field(default_factory=list)

    def profile_search(
        self,
        faiss_index: faiss.Index,
        duckdb_connection,
        query_vector: np.ndarray,
        n_neighbors: int = 1000,
        nprobe: int = 4096,
    ) -> SearchProfile:
        """Profile a single search operation."""
        profile = SearchProfile()
        total_start = time.perf_counter()

        # 1. FAISS search
        faiss_start = time.perf_counter()
        query_np = query_vector.reshape(1, -1).astype("float32")
        params = faiss.SearchParametersIVF(nprobe=nprobe)
        distances, ids = faiss_index.search(query_np, n_neighbors, params=params)
        faiss_ids = ids[0].tolist()
        profile.faiss_search_ms = (time.perf_counter() - faiss_start) * 1000

        # 2. DuckDB metadata query
        duckdb_start = time.perf_counter()
        if faiss_ids:
            placeholders = ",".join(["?" for _ in faiss_ids])
            sql = f"""
            SELECT id, ST_AsGeoJSON(geometry) AS geometry_json
            FROM geo_embeddings
            WHERE id IN ({placeholders})
            """
            result = duckdb_connection.execute(sql, faiss_ids).fetchdf()
            profile.n_results = len(result)
        profile.duckdb_query_ms = (time.perf_counter() - duckdb_start) * 1000

        profile.total_ms = (time.perf_counter() - total_start) * 1000
        self.profiles.append(profile)
        return profile

    def profile_duckdb_only(
        self,
        duckdb_connection,
        ids: list,
    ) -> float:
        """Profile just the DuckDB query with given IDs."""
        start = time.perf_counter()
        placeholders = ",".join(["?" for _ in ids])
        sql = f"""
        SELECT id, ST_AsGeoJSON(geometry) AS geometry_json
        FROM geo_embeddings
        WHERE id IN ({placeholders})
        """
        duckdb_connection.execute(sql, ids).fetchdf()
        elapsed_ms = (time.perf_counter() - start) * 1000
        return elapsed_ms

    def summary(self) -> str:
        """Print summary of all profiles."""
        if not self.profiles:
            return "No profiles recorded."

        lines = ["Search Profile Summary:", "=" * 50]
        for i, p in enumerate(self.profiles):
            lines.append(f"\nSearch #{i + 1}:")
            lines.append(f"  FAISS:  {p.faiss_search_ms:>8.1f} ms")
            lines.append(f"  DuckDB: {p.duckdb_query_ms:>8.1f} ms")
            lines.append(f"  Total:  {p.total_ms:>8.1f} ms")

        if len(self.profiles) > 1:
            lines.append("\n" + "-" * 50)
            lines.append("Comparison (first vs subsequent):")
            first = self.profiles[0]
            rest = self.profiles[1:]
            avg_rest_faiss = sum(p.faiss_search_ms for p in rest) / len(rest)
            avg_rest_duckdb = sum(p.duckdb_query_ms for p in rest) / len(rest)
            lines.append(
                f"  FAISS:  {first.faiss_search_ms:.1f} ms -> {avg_rest_faiss:.1f} ms "
                f"({first.faiss_search_ms / avg_rest_faiss:.1f}x speedup)"
            )
            lines.append(
                f"  DuckDB: {first.duckdb_query_ms:.1f} ms -> {avg_rest_duckdb:.1f} ms "
                f"({first.duckdb_query_ms / avg_rest_duckdb:.1f}x speedup)"
            )

        return "\n".join(lines)

    def clear(self):
        """Clear recorded profiles."""
        self.profiles = []


def run_profile_test(app, n_searches: int = 5):
    """Run profiling test on a GeoVibes app instance.

    Usage in notebook:
        from geovibes.database.profile_search import run_profile_test
        run_profile_test(app, n_searches=5)
    """
    if app.state.query_vector is None:
        print("Error: No query vector. Label some points first.")
        return

    profiler = SearchProfiler()
    print(f"Running {n_searches} searches to measure caching effect...\n")

    for i in range(n_searches):
        profile = profiler.profile_search(
            faiss_index=app.data.faiss_index,
            duckdb_connection=app.data.duckdb_connection,
            query_vector=app.state.query_vector,
            n_neighbors=1000,
        )
        print(
            f"Search #{i + 1}: FAISS={profile.faiss_search_ms:.0f}ms, DuckDB={profile.duckdb_query_ms:.0f}ms, Total={profile.total_ms:.0f}ms"
        )

    print("\n" + profiler.summary())
    return profiler
