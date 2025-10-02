"""Contextual bandit environment for docking recipe selection.

This module exposes a Gymnasium-compatible one-step environment whose reward
function is computed by running docking engines (QuickVina2GPU / Glide).
It mirrors the high-level design sketched in the user instructions and adds
thin wrappers around the existing docking utilities in ``gpuvina.py`` and
``glide_dock.py`` so they can be called from reinforcement-learning code.

The environment returns context features as observations, accepts discrete
actions corresponding to docking recipes, and computes rewards as a weighted
combination of BEDROC_\alpha early-recognition score and the wall-clock time
needed to execute the docking recipe.

The code has three main building blocks:

``DockingBanditEnv``
    Implements the Gymnasium ``Env`` API for a one-step contextual bandit.

``DockingBackend``
    Orchestrates calls into the docking engines and converts their outputs into
    rewards by computing BEDROC and applying the time penalty.

``VinaRunner`` and ``GlideRunner``
    Minimal adapters over ``QuickVina2GPU`` / ``get_vina_scores_mul_gpu`` and
    ``get_glide_scores_mul_gpu`` respectively.  They keep the interface simple
    for the bandit while allowing the caller to decide whether to run jobs
    locally (QuickVina2GPU) or through the Slurm-based multi-GPU helpers.

See the ``__main__`` block for an executable example that mocks the docking
backends so the environment can be stepped through without requiring access to
Schrödinger or a GPU cluster.  To plug in the real docking engines simply
instantiate ``VinaRunner`` / ``GlideRunner`` with the appropriate configuration
objects and pass them to ``DockingBackend``.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gpuvina import QuickVina2GPU, get_vina_scores_mul_gpu
from glide_dock import get_glide_scores_mul_gpu


ArrayLike = Sequence[float]


def compute_bedroc(labels: ArrayLike, scores: ArrayLike, alpha: float = 20.0) -> float:
    """Compute the BEDROC_α early enrichment metric."""

    labels_arr = np.asarray(labels, dtype=float)
    scores_arr = np.asarray(scores, dtype=float)

    if labels_arr.ndim != 1 or scores_arr.ndim != 1:
        raise ValueError("Labels and scores must be 1-D arrays")
    if labels_arr.shape[0] != scores_arr.shape[0]:
        raise ValueError("Labels and scores must have the same length")

    n_total = labels_arr.size
    n_actives = float(labels_arr.sum())
    if n_actives == 0:
        return 0.0
    if n_actives == n_total:
        return 1.0

    order = np.argsort(-scores_arr)
    ranked_labels = labels_arr[order]
    active_ranks = np.nonzero(ranked_labels)[0] + 1  # 1-based ranks

    frac_actives = n_actives / n_total
    term = np.exp(-alpha * active_ranks / n_total).sum()

    # Constants from Truchon & Bayly, J. Chem. Inf. Model. 2007, 47, 488–508
    c1 = alpha / (1.0 - np.exp(-alpha))
    c2 = 1.0 / n_actives
    c3 = np.exp(-alpha * frac_actives)
    c4 = 1.0 - np.exp(-alpha * (1.0 - frac_actives))

    bedroc = c1 * c2 * term
    bedroc = bedroc / c4
    bedroc += (1.0 - c3) / (1.0 - np.exp(-alpha))
    bedroc -= frac_actives
    bedroc *= (1.0 - np.exp(-alpha))
    bedroc += frac_actives

    return float(np.clip(bedroc, 0.0, 1.0))


@dataclass(frozen=True)
class DockingRecipe:
    """Description of a docking recipe that can be chosen by the agent."""

    name: str
    engine: str  # e.g. "vina" or "glide"
    params: Mapping[str, Any] = dataclasses.field(default_factory=dict)


@dataclass
class DockingContext:
    """Context describing the bandit round."""

    features: np.ndarray
    smiles: Sequence[str]
    labels: np.ndarray
    split: str = "train"
    metadata: Dict[str, Any] = field(default_factory=dict)


class DockingContextSampler:
    """Iterator-style helper that serves contextual bandit rounds."""

    def __init__(self, contexts: Iterable[DockingContext]):
        self._contexts = list(contexts)
        if not self._contexts:
            raise ValueError("At least one context must be provided")
        self._rng = np.random.default_rng()

    def sample(self) -> DockingContext:
        idx = self._rng.integers(0, len(self._contexts))
        return self._contexts[idx]


class VinaRunner:
    """Adapter for running GPU Vina either locally or via the Slurm helper."""

    def __init__(
        self,
        quick_vina_kwargs: Optional[Mapping[str, Any]] = None,
        *,
        cursor: Any = None,
        config: Any = None,
        num_gpus: int = 1,
        output_dir: str = "vina_results",
    ) -> None:
        self._quick_vina: Optional[QuickVina2GPU] = None
        if quick_vina_kwargs:
            self._quick_vina = QuickVina2GPU(**quick_vina_kwargs)
        self._cursor = cursor
        self._config = config
        self._num_gpus = num_gpus
        self._output_dir = output_dir

    def dock(self, smiles: Sequence[str], **overrides: Any) -> Tuple[np.ndarray, float]:
        start = time.perf_counter()
        if self._quick_vina is not None:
            _, scores = self._quick_vina.dock_mols(list(smiles))
            runtime = (time.perf_counter() - start) / 60.0
            return np.asarray(scores, dtype=float), runtime

        config = overrides.get("config", self._config)
        if config is None:
            raise RuntimeError(
                "VinaRunner requires either quick_vina_kwargs or a config for get_vina_scores_mul_gpu"
            )
        cursor = overrides.get("cursor", self._cursor)
        num_gpus = overrides.get("num_gpus", self._num_gpus)
        output_dir = overrides.get("output_dir", self._output_dir)
        dockscore_gt = overrides.get("dockscore_gt")
        scores = get_vina_scores_mul_gpu(
            list(smiles), cursor, config, num_gpus=num_gpus, output_dir=output_dir, dockscore_gt=dockscore_gt
        )
        runtime = (time.perf_counter() - start) / 60.0
        return np.asarray(scores, dtype=float), runtime


class GlideRunner:
    """Adapter over the Glide Slurm helper."""

    def __init__(self, config: Any, *, output_dir: str = "glide_results") -> None:
        self._config = config
        self._output_dir = output_dir

    def dock(
        self,
        batches: Mapping[str, Any],
        *,
        iteration: int = 0,
        dockscore_gt: Optional[Mapping[str, float]] = None,
        output_dir: Optional[str] = None,
    ) -> Tuple[Dict[str, np.ndarray], float]:
        start = time.perf_counter()
        train_scores, val_scores, test_scores = get_glide_scores_mul_gpu(
            batches,
            iteration,
            self._config,
            output_dir=output_dir or self._output_dir,
            dockscore_gt=dockscore_gt,
        )
        runtime = (time.perf_counter() - start) / 60.0
        return {
            "train": np.asarray(train_scores, dtype=float),
            "validation": np.asarray(val_scores, dtype=float),
            "test": np.asarray(test_scores, dtype=float),
        }, runtime


class DockingBackend:
    """Turn docking results into contextual bandit rewards."""

    def __init__(
        self,
        *,
        vina_runner: Optional[VinaRunner] = None,
        glide_runner: Optional[GlideRunner] = None,
        reward_weights: Tuple[float, float] = (1.0, 0.02),
        bedroc_alpha: float = 20.0,
    ) -> None:
        self._vina_runner = vina_runner
        self._glide_runner = glide_runner
        self._w_bedroc, self._w_time = reward_weights
        self._bedroc_alpha = bedroc_alpha

    def evaluate(self, recipe: DockingRecipe, context: DockingContext) -> Tuple[float, Dict[str, Any]]:
        if recipe.engine == "vina":
            if self._vina_runner is None:
                raise RuntimeError("No VinaRunner configured")
            scores, runtime = self._vina_runner.dock(context.smiles, **recipe.params)
        elif recipe.engine == "glide":
            if self._glide_runner is None:
                raise RuntimeError("No GlideRunner configured")
            batches = context.metadata.get("glide_batches")
            if batches is None:
                raise ValueError("Glide recipe requires 'glide_batches' in the context metadata")
            iteration = context.metadata.get("iteration", 0)
            scores_dict, runtime = self._glide_runner.dock(
                batches,
                iteration=iteration,
                dockscore_gt=recipe.params.get("dockscore_gt"),
                output_dir=recipe.params.get("output_dir"),
            )
            split = context.split
            if split not in scores_dict:
                raise ValueError(f"Unknown split '{split}' in Glide results")
            scores = scores_dict[split]
        else:
            raise ValueError(f"Unsupported docking engine '{recipe.engine}'")

        labels = context.labels
        if labels.shape[0] != scores.shape[0]:
            raise ValueError(
                "Number of docking scores does not match number of labels "
                f"({scores.shape[0]} vs {labels.shape[0]})"
            )

        bedroc = compute_bedroc(labels, -scores, alpha=self._bedroc_alpha)
        reward = self._w_bedroc * bedroc - self._w_time * runtime
        info = {
            "recipe": recipe.name,
            "engine": recipe.engine,
            "bedroc": bedroc,
            "runtime_min": runtime,
            "scores": scores,
        }
        return reward, info


class DockingBanditEnv(gym.Env):
    """One-step contextual bandit that chooses between docking recipes."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        recipes: Sequence[DockingRecipe],
        context_sampler: DockingContextSampler,
        backend: DockingBackend,
        ctx_dim: int,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.recipes = list(recipes)
        if not self.recipes:
            raise ValueError("At least one docking recipe must be provided")
        self.context_sampler = context_sampler
        self.backend = backend
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(ctx_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.recipes))
        self._rng = np.random.default_rng(seed)
        self._current_context: Optional[DockingContext] = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        context = self.context_sampler.sample()
        self._current_context = context
        observation = context.features.astype(np.float32)
        return observation, {}

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} outside of space {self.action_space}")
        if self._current_context is None:
            raise RuntimeError("Environment must be reset before stepping")

        recipe = self.recipes[action]
        reward, info = self.backend.evaluate(recipe, self._current_context)

        terminated = True  # one-step bandit
        truncated = False
        observation = self._current_context.features.astype(np.float32)
        return observation, reward, terminated, truncated, info


class RandomDockingRunner:
    """Fallback runner used in the example when real docking is unavailable."""

    def __init__(self, engine: str, rng: Optional[np.random.Generator] = None) -> None:
        self.engine = engine
        self._rng = rng or np.random.default_rng(0)

    def dock(self, payload: Any, **_: Any):  # type: ignore[override]
        if self.engine == "glide":
            size = len(payload["train"].libID) if hasattr(payload["train"], "libID") else len(payload["train"])
            scores = {
                "train": self._rng.normal(-8.0, 1.0, size=size),
                "validation": self._rng.normal(-8.0, 1.0, size=size),
                "test": self._rng.normal(-8.0, 1.0, size=size),
            }
        else:
            size = len(payload)
            scores = self._rng.normal(-8.0, 1.0, size=size)
        runtime = self._rng.uniform(3.0, 15.0) / 60.0  # minutes
        return scores, runtime


def build_demo_environment(ctx_dim: int = 12) -> DockingBanditEnv:
    rng = np.random.default_rng(42)
    smiles = ["CCO", "CCN", "CCCl", "c1ccccc1"]
    labels = rng.integers(0, 2, size=len(smiles)).astype(float)
    features = rng.normal(size=(ctx_dim,))
    contexts = [
        DockingContext(features=features, smiles=smiles, labels=labels, split="train")
    ]
    sampler = DockingContextSampler(contexts)

    vina_backend = RandomDockingRunner("vina", rng)
    glide_backend = RandomDockingRunner("glide", rng)

    backend = DockingBackend(
        vina_runner=vina_backend,  # type: ignore[arg-type]
        glide_runner=glide_backend,  # type: ignore[arg-type]
        reward_weights=(1.0, 0.02),
    )

    recipes = [
        DockingRecipe(name="Vina_fast", engine="vina"),
        DockingRecipe(name="Glide_standard", engine="glide"),
    ]
    env = DockingBanditEnv(recipes, sampler, backend, ctx_dim=ctx_dim)
    return env


if __name__ == "__main__":
    env = build_demo_environment()
    obs, _ = env.reset()
    print("Initial context:", obs)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Action {action} -> reward {reward:.3f}, BEDROC {info['bedroc']:.3f}, runtime {info['runtime_min']:.2f} min")
    print("Episode done?", terminated or truncated)

