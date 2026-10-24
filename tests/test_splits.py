import json
import random

import pytest

from codegen.data.apps import Problem
from codegen.data.splits import (
    FIXED_SPLITS, SplitConfig, build_splits, load_split, write_splits,
)

WORDS = [f"w{i}" for i in range(5000)]
APPS_DIFFS = ["introductory", "interview", "competition"]
TACO_DIFFS = ["EASY", "MEDIUM", "HARD"]
TRAINING = FIXED_SPLITS + ("grpo_train",)


def text(rng):
    return " ".join(rng.choice(WORDS) for _ in range(60))


def problem(uid, difficulty, kind="stdio", question="", n_solutions=1, n_tests=1):
    split, i = uid.rsplit("-", 1)
    return Problem(
        uid=uid, source_split=split, source_id=int(i), difficulty=difficulty,
        question=question, starter_code="", fn_name="f" if kind == "call" else None,
        inputs=["1"] * n_tests, outputs=["1"] * n_tests, n_solutions=n_solutions, url="",
    )


@pytest.fixture
def data():
    rng = random.Random(1)
    # Like real APPS-train, most program-style problems have a single test.
    train = [
        problem(f"train-{i:04d}", APPS_DIFFS[i % 3], "call" if i % 2 else "stdio", text(rng),
                n_tests=5 if i % 5 == 0 else 1)
        for i in range(600)
    ]
    test = [
        problem(f"test-{i:04d}", APPS_DIFFS[i % 3], question=text(rng),
                n_solutions=0 if i % 10 == 0 else 1, n_tests=10)
        for i in range(300)
    ]
    taco = [
        problem(f"taco-{i:05d}", TACO_DIFFS[i % 3], "call" if i % 4 == 3 else "stdio",
                text(rng), n_tests=1 if i % 7 == 0 else 50)
        for i in range(300)
    ]
    # train-0020 (interview, 5 tests) copies test-0001.
    train[20] = problem("train-0020", "interview", question=test[1].question, n_tests=5)
    # taco-00002 copies test-0290, which is unlikely to be sampled into eval:
    # copies of any APPS-test problem are banned, not just eval ones.
    taco[2] = problem("taco-00002", "EASY", question=test[290].question, n_tests=50)
    # taco-00004 copies train-0010 (introductory, stdio, 5 tests) with more tests.
    taco[4] = problem("taco-00004", "MEDIUM", question=train[10].question, n_tests=50)
    return train, test, taco


CFG = SplitConfig(
    eval_counts={"introductory": 20, "interview": 40, "competition": 10},
    sizes={"dev": 50, "gepa_val": 30, "gepa_train": 40},
)


def run(data, cfg=CFG):
    train, test, taco = data
    return build_splits(train, test, taco, cfg)


def by_uid(data):
    return {p.uid: p for ps in data for p in ps}


def placed(r):
    return {u: name for name, ids in r.ids.items() for u in ids}


def test_sizes_disjoint_and_filtered(data):
    r = run(data)
    assert len(r.ids["eval"]) == 70
    for name, n in CFG.sizes.items():
        assert len(r.ids[name]) == n
    all_ids = [u for ids in r.ids.values() for u in ids]
    assert len(all_ids) == len(set(all_ids))

    p = by_uid(data)
    assert all(p[u].n_solutions > 0 for u in r.ids["eval"])
    training = [u for name in TRAINING for u in r.ids[name]]
    assert all(p[u].test_kind == "stdio" for u in training)
    assert all(p[u].n_tests >= CFG.min_tests for u in training)
    assert all(p[u].difficulty not in ("competition", "HARD") for u in training)
    assert {p[u].source_split for u in training} == {"train", "taco"}


def test_single_test_problems_excluded(data):
    r = run(data)
    where = placed(r)
    assert "train-0002" not in where  # interview, stdio, 1 test
    assert "taco-00007" not in where  # MEDIUM, stdio, 1 test


def test_copies_of_any_apps_test_problem_dropped(data):
    r = run(data)
    assert "test-0290" not in r.ids["eval"]
    rows = r.dropped["dup_of_apps_test"]
    assert ("taco-00002", "test-0290", 1.0) in rows
    assert ("train-0020", "test-0001", 1.0) in rows
    assert "taco-00002" not in placed(r) and "train-0020" not in placed(r)


def test_cross_dataset_duplicate_keeps_copy_with_most_tests(data):
    r = run(data)
    assert "taco-00004" in placed(r)
    assert "train-0010" not in placed(r)
    assert ("train-0010", "taco-00004", 1.0) in r.dropped["dup_within_pool"]


def test_deterministic_and_seed_sensitive(data):
    assert run(data).ids == run(data).ids
    other = SplitConfig(**{**CFG.__dict__, "seed": 1})
    assert run(data, other).ids["eval"] != run(data).ids["eval"]


def test_eval_independent_of_train_config(data):
    bigger_dev = SplitConfig(**{**CFG.__dict__, "sizes": {**CFG.sizes, "dev": 80}})
    assert run(data).ids["eval"] == run(data, bigger_dev).ids["eval"]


def test_pool_too_small_raises(data):
    huge = SplitConfig(**{**CFG.__dict__, "sizes": {**CFG.sizes, "dev": 10_000}})
    with pytest.raises(ValueError, match="pool has"):
        run(data, huge)


def test_write_freeze_guard_and_load_roundtrip(data, tmp_path):
    r = run(data)
    problems = [p for ps in data for p in ps]
    splits_dir, processed = tmp_path / "splits", tmp_path / "processed"
    write_splits(r, CFG, problems, splits_dir, force=False, processed_dir=processed)
    write_splits(r, CFG, problems, splits_dir, force=False, processed_dir=processed)  # same eval: OK

    manifest = json.loads((splits_dir / "manifest.json").read_text())
    assert manifest["splits"]["eval"]["n"] == 70
    loaded = load_split("dev", splits_dir, processed)
    assert [p.uid for p in loaded] == r.ids["dev"]
    assert loaded[0] == by_uid(data)[loaded[0].uid]

    r.ids["eval"] = r.ids["eval"][1:]
    with pytest.raises(SystemExit):
        write_splits(r, CFG, problems, splits_dir, force=False)
