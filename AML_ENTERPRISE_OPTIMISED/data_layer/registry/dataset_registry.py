"""data_layer/registry/dataset_registry.py"""
from __future__ import annotations
import pandas as pd
from config.job_types import JOB_REGISTRY

class DatasetRegistry:
    def __init__(self):
        self._store = {}
        self._ctry_df = None
        self._loaded = False

    def load_all(self, n_san_ind=5000, n_san_ent=3000, n_scion=2000, n_pep=2000,
                 n_nns_art=3000, n_nns_str=2000, seed=42, verbose=True):
        from data_layer.generators.san_individual import generate as g1
        from data_layer.generators.san_entity     import generate as g2
        from data_layer.generators.scion          import generate as g3
        from data_layer.generators.pep            import generate as g4
        from data_layer.generators.nns            import generate_articles as g5, generate_structured as g6
        from data_layer.generators.ctry           import generate as g7
        for label, fn, kw in [
            ("SAN INDIVIDUAL", g1, {"n":n_san_ind,"seed":seed}),
            ("SAN ENTITY",     g2, {"n":n_san_ent,"seed":seed}),
            ("SCION",          g3, {"n":n_scion,  "seed":seed}),
            ("PEP",            g4, {"n":n_pep,    "seed":seed}),
            ("NNS ARTICLES",   g5, {"n":n_nns_art,"seed":seed}),
            ("NNS STRUCTURED", g6, {"n":n_nns_str,"seed":seed}),
        ]:
            if verbose: print(f"[Registry] Loading {label:20s} ...", end=" ", flush=True)
            df = fn(**kw); self._ingest(df)
            if verbose: print(f"{len(df):,} records")
        if verbose: print("[Registry] Loading COUNTRY RISK REGISTER ...", end=" ", flush=True)
        self._ctry_df = g7()
        if verbose: print(f"{len(self._ctry_df)} countries")
        self._loaded = True
        if verbose: print(f"\n[Registry] Ready: {self.total_records():,} name-match + {len(self._ctry_df)} country records\n")
        return self

    def _ingest(self, df):
        for (ds, etype), part in df.groupby(["dataset_type","entity_type"]):
            key = (str(ds).upper(), str(etype).upper())
            if key in self._store:
                self._store[key] = pd.concat([self._store[key], part], ignore_index=True)
            else:
                self._store[key] = part.reset_index(drop=True)

    def get(self, dataset_code, entity_type=None):
        code = dataset_code.upper()
        if entity_type:
            return self._store.get((code, entity_type.upper()), pd.DataFrame())
        parts = [df for (c,_),df in self._store.items() if c==code]
        return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()

    def get_country_register(self):
        return self._ctry_df if self._ctry_df is not None else pd.DataFrame()

    def get_for_job(self, job_code):
        jc = job_code.strip().upper()
        if jc not in JOB_REGISTRY: raise ValueError(f"Unknown job '{jc}'")
        job = JOB_REGISTRY[jc]
        results = []
        for ds_cfg in sorted(job.datasets, key=lambda x: x.priority):
            for etype in ds_cfg.entity_types:
                df = self.get(ds_cfg.code, entity_type=etype)
                if not df.empty: results.append((ds_cfg.code, df))
        return results

    def list_partitions(self):
        return [(c,e,len(df)) for (c,e),df in sorted(self._store.items())]

    def total_records(self):
        return sum(len(df) for df in self._store.values())

    def summary(self):
        lines = ["DatasetRegistry -- 10-Job Schema","="*55]
        for c,e,n in self.list_partitions():
            lines.append(f"  {c:<22} {e:<12} {n:>8,} records")
        if self._ctry_df is not None:
            lines.append(f"  {'COUNTRY_RISK':<22} {'LOOKUP':<12} {len(self._ctry_df):>8} countries")
        lines.append("-"*55)
        lines.append(f"  {'TOTAL':<34} {self.total_records():>8,} records")
        return "\n".join(lines)

    def list_datasets(self):
        """Alias for list_partitions — used by MultiDatasetIndex."""
        return self.list_partitions()

