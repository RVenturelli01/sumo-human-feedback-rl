"""The aggregator must recognise every directory a run can create.

`make_run_dir` appends `_01` when the name is already taken, so a re-run of a
cell lands in a directory the plain `<campaign>_<arm>_B<budget>-seed<n>` pattern
does not match. If it goes unmatched the aggregate silently keeps reading the
older copy.
"""
import evaluate


def match(name):
    m = evaluate.RUN_DIR_RE.fullmatch(name)
    return m.groupdict() if m else None


def test_plain_name():
    assert match("main_hybrid_soft_B10-seed1") == {
        "campaign": "main", "arm": "hybrid_soft", "budget": "10", "seed": "1"}


def test_repeat_run_resolves_to_the_same_cell():
    plain = match("th_1mh4iq5_unw_bern_B10-seed10")
    assert match("th_1mh4iq5_unw_bern_B10-seed10_02") == plain


def test_repeats_are_reported_as_duplicated_seeds():
    rows = [{"method": "Hybrid-soft", "budget": 10, "seed": 1, "run_name": n}
            for n in ("main_hybrid_soft_B10-seed1", "main_hybrid_soft_B10-seed1_01")]
    problems = evaluate.check_grid(rows)
    assert any("duplicated seeds [1]" in p for p in problems)


def test_unrelated_names_are_ignored():
    assert match("multirun") is None
    assert match("main_hybrid_soft_B10-seed1_0") is None
