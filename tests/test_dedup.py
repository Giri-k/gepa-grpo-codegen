from codegen.data.dedup import clusters, near_duplicate_pairs

BASE = (
    "Vasya has an array of n integers and wants to know the maximum sum of a "
    "contiguous subarray whose length is at most k. Print the answer modulo a prime."
)


def test_finds_near_duplicate_ignores_unrelated():
    texts = {
        "a": BASE,
        "b": BASE.replace("Vasya", "Petya"),  # renamed character, otherwise identical
        "c": "Given a tree with n vertices, count paths whose edge weights sum to zero.",
    }
    pairs = near_duplicate_pairs(texts, threshold=0.6)
    assert [(a, b) for a, b, _ in pairs] == [("a", "b")]


def test_whitespace_and_case_insensitive():
    texts = {"a": BASE, "b": "  " + BASE.upper().replace(" ", "\n  ")}
    (a, b, score), = near_duplicate_pairs(texts)
    assert score == 1.0


def test_latex_and_unicode_versions_match():
    # Real pair from APPS-test (LaTeX) and TACO (Unicode), which scored 0.39 before
    # formatting was stripped.
    apps_text = (
        "You are given $n$ positive integers $a_1, \\ldots, a_n$, and an integer $k \\geq 2$. "
        "Count the number of pairs $i, j$ such that $1 \\leq i < j \\leq n$, and there exists "
        "an integer $x$ such that $a_i \\cdot a_j = x^k$."
    )
    taco_text = (
        "You are given n positive integers a_1, …, a_n, and an integer k ≥ 2. Count the number "
        "of pairs i, j such that 1 ≤ i < j ≤ n, and there exists an integer x such that "
        "a_i ⋅ a_j = x^k."
    )
    (_, _, score), = near_duplicate_pairs({"apps": apps_text, "taco": taco_text}, threshold=0.0)
    assert score == 1.0


def test_image_placeholder_ignored():
    texts = {"a": BASE + " Formula: <image> holds.", "b": BASE + " Formula: holds."}
    (_, _, score), = near_duplicate_pairs(texts, threshold=0.0)
    assert score == 1.0


def test_clusters_are_connected_components():
    groups = clusters(["a", "b", "c", "d"], [("a", "b", 1.0), ("b", "c", 0.9)])
    assert groups == [["a", "b", "c"], ["d"]]
