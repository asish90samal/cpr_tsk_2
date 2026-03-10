"""
blocking_engine/inverted_index.py
───────────────────────────────────
InvertedIndex — replaces pandas full-scan blocking for million-scale datasets.

Instead of scanning all rows on every query, the index is built ONCE at load
time and lookups run in O(tokens_in_query) time regardless of dataset size.

Architecture
------------
  _token_index : dict[str, set[int]]
      Maps every normalised name token → set of row positions in the DataFrame.
      Built from primary_name + all aliases.

  _phonetic_index : dict[str, set[int]]
      Maps Double Metaphone codes → row positions.

  _id_index : dict[str, int]
      Maps exact identifier strings (passport, IMO, account number, etc.)
      → row position. Used for O(1) exact-ID lookup.

Usage
-----
    idx = InvertedIndex()
    idx.build(df)                          # build once at startup

    candidates = idx.query("MOHAMMED HUSSAIN KHAN")   # token + phonetic
    exact_hit  = idx.exact_id_lookup("P1234567")      # O(1) ID lookup
"""

from __future__ import annotations
import re
from collections import defaultdict

import pandas as pd

try:
    from metaphone import doublemetaphone
    _METAPHONE_OK = True
except ImportError:
    _METAPHONE_OK = False

# Tokens shorter than this are ignored (too common / noisy)
_MIN_TOKEN_LEN = 3

# Columns that may carry exact identifiers (checked in order)
_ID_COLUMNS = [
    "id_number", "registration_number", "account_number",
    "imo_number", "mmsi_number", "tail_number", "icao_number",
    "passport_number",
]


def _tokenise(name: str) -> list[str]:
    """Normalise and split a name into meaningful tokens."""
    name = name.upper().strip()
    name = re.sub(r"[^\w\s]", " ", name)
    return [t for t in name.split() if len(t) >= _MIN_TOKEN_LEN]


def _metaphone_codes(token: str) -> list[str]:
    if not _METAPHONE_OK:
        return []
    codes = doublemetaphone(token)
    return [c for c in codes if c]


class InvertedIndex:
    """
    Token-based inverted index over a pandas DataFrame.

    Provides sub-millisecond candidate retrieval for datasets of any size
    without loading the full DataFrame into working memory on each query.
    """

    def __init__(self) -> None:
        self._df: pd.DataFrame | None       = None
        self._token_index:    dict[str, set[int]] = defaultdict(set)
        self._phonetic_index: dict[str, set[int]] = defaultdict(set)
        self._id_index:       dict[str, int]      = {}
        self._built: bool = False

    # ── Build ──────────────────────────────────────────────────────────────

    def build(self, df: pd.DataFrame, verbose: bool = False) -> "InvertedIndex":
        """
        Build the index from a DataFrame.
        Indexes primary_name, all aliases, and all ID columns.

        Parameters
        ----------
        df      : The full sanctions/watchlist DataFrame
        verbose : Print progress
        """
        self._df            = df.reset_index(drop=True)
        self._token_index   = defaultdict(set)
        self._phonetic_index = defaultdict(set)
        self._id_index      = {}

        if verbose:
            print(f"[InvertedIndex] Building index for {len(df):,} records...")

        for row_idx, row in df.iterrows():
            # ── Collect all name strings for this row ──────────────────
            names: list[str] = []
            if pd.notna(row.get("primary_name", "")):
                names.append(str(row["primary_name"]))
            aliases_str = row.get("aliases", "")
            if pd.notna(aliases_str) and aliases_str:
                names.extend(str(aliases_str).split("|"))

            # ── Token index + phonetic index ───────────────────────────
            for name in names:
                for token in _tokenise(name):
                    self._token_index[token].add(row_idx)
                    for code in _metaphone_codes(token):
                        self._phonetic_index[code].add(row_idx)

            # ── Exact ID index ─────────────────────────────────────────
            for col in _ID_COLUMNS:
                val = row.get(col, "")
                if pd.notna(val) and str(val).strip():
                    self._id_index[str(val).strip().upper()] = row_idx

        self._built = True
        if verbose:
            print(f"[InvertedIndex] Done — {len(self._token_index):,} token keys, "
                  f"{len(self._phonetic_index):,} phonetic keys, "
                  f"{len(self._id_index):,} ID keys")
        return self

    # ── Query ──────────────────────────────────────────────────────────────

    def query(
        self,
        name: str,
        use_phonetic: bool = True,
        min_token_hits: int = 1,
    ) -> pd.DataFrame:
        """
        Retrieve candidate rows matching the query name.

        A row is included if it shares at least `min_token_hits` tokens
        (or phonetic codes) with the query name.

        Parameters
        ----------
        name            : Normalised query name string
        use_phonetic    : Also include phonetically similar candidates
        min_token_hits  : Minimum token overlap required (default 1)

        Returns
        -------
        pd.DataFrame — subset of the indexed DataFrame
        """
        self._assert_built()
        tokens = _tokenise(name)
        if not tokens:
            return self._df.iloc[0:0]

        # Count token hits per row
        hit_counts: dict[int, int] = defaultdict(int)
        for token in tokens:
            for row_idx in self._token_index.get(token, set()):
                hit_counts[row_idx] += 1

        # Phonetic hits (count separately — any phonetic hit qualifies)
        if use_phonetic and _METAPHONE_OK:
            for token in tokens:
                for code in _metaphone_codes(token):
                    for row_idx in self._phonetic_index.get(code, set()):
                        hit_counts[row_idx] = max(hit_counts[row_idx], min_token_hits)

        # Filter by minimum hits
        candidate_idxs = [
            idx for idx, count in hit_counts.items()
            if count >= min_token_hits
        ]

        if not candidate_idxs:
            return self._df.iloc[0:0]

        return self._df.iloc[sorted(candidate_idxs)].copy().reset_index(drop=True)

    def exact_id_lookup(self, identifier: str) -> pd.DataFrame | None:
        """
        O(1) exact identifier lookup (passport, IMO number, account number, etc.).

        Returns
        -------
        Single-row DataFrame if found, else None.
        """
        self._assert_built()
        key = identifier.strip().upper()
        row_idx = self._id_index.get(key)
        if row_idx is None:
            return None
        return self._df.iloc[[row_idx]].copy().reset_index(drop=True)

    def query_with_id(
        self,
        name: str,
        identifier: str | None = None,
        use_phonetic: bool = True,
    ) -> tuple[pd.DataFrame, bool]:
        """
        Combined query: exact ID lookup first (returns auto-ALERT flag),
        then name-based token query for remaining candidates.

        Returns
        -------
        (candidates_df, exact_id_hit)
            exact_id_hit=True means an identifier-matched row was found
            (should trigger auto-ALERT per DatasetConfig.auto_alert_on_exact_id)
        """
        exact_hit  = False
        exact_rows = pd.DataFrame()

        if identifier:
            exact_result = self.exact_id_lookup(identifier)
            if exact_result is not None:
                exact_hit  = True
                exact_rows = exact_result

        name_candidates = self.query(name, use_phonetic=use_phonetic)

        # Merge: exact hit rows + name candidates, deduplicate
        if not exact_rows.empty:
            combined = pd.concat([exact_rows, name_candidates], ignore_index=True)
            if "entity_id" in combined.columns:
                combined = combined.drop_duplicates(subset=["entity_id"])
            candidates = combined.reset_index(drop=True)
        else:
            candidates = name_candidates

        return candidates, exact_hit

    # ── Multi-dataset query ────────────────────────────────────────────────

    def stats(self) -> dict[str, int]:
        """Return index statistics."""
        self._assert_built()
        return {
            "indexed_rows":    len(self._df) if self._df is not None else 0,
            "token_keys":      len(self._token_index),
            "phonetic_keys":   len(self._phonetic_index),
            "id_keys":         len(self._id_index),
        }

    def _assert_built(self) -> None:
        if not self._built or self._df is None:
            raise RuntimeError("Index not built. Call .build(df) first.")


class MultiDatasetIndex:
    """
    Holds one InvertedIndex per (dataset_code, entity_type) partition.
    Provides a single fan-out query across all relevant datasets.

    Usage
    -----
        multi_idx = MultiDatasetIndex()
        multi_idx.build_from_registry(registry)

        results = multi_idx.query_all(
            name="MOHAMMED HUSSAIN",
            entity_type="INDIVIDUAL",
            job_type="INDIVIDUAL_JOB",
        )
    """

    def __init__(self) -> None:
        self._indexes: dict[tuple[str, str], InvertedIndex] = {}

    def build_from_registry(
        self,
        registry,   # DatasetRegistry — avoid circular import with string type
        verbose: bool = True,
    ) -> "MultiDatasetIndex":
        """Build one index per (dataset_code, entity_type) partition."""
        for code, etype, count in registry.list_datasets():
            if count == 0:
                continue
            df  = registry.get(code, entity_type=etype)
            key = (code, etype)
            if verbose:
                print(f"[MultiDatasetIndex] Indexing {code} / {etype} ({count:,} rows)...")
            idx = InvertedIndex()
            idx.build(df, verbose=False)
            self._indexes[key] = idx

        if verbose:
            print(f"[MultiDatasetIndex] {len(self._indexes)} indexes built.")
        return self

    def query_dataset(
        self,
        dataset_code: str,
        entity_type:  str,
        name:         str,
        identifier:   str | None = None,
    ) -> tuple[pd.DataFrame, bool]:
        """Query a single dataset index."""
        key = (dataset_code.upper(), entity_type.upper())
        idx = self._indexes.get(key)
        if idx is None:
            return pd.DataFrame(), False
        return idx.query_with_id(name, identifier=identifier)

    def list_indexes(self) -> list[tuple[str, str]]:
        return list(self._indexes.keys())
