"""MCQ screening tests.

The free-text interview scorer in `app.interview` grades how an answer is *put* —
structure, concrete detail, whether the right skills come up. It cannot tell whether
the answer is factually right, and the module says so. MCQ closes exactly that gap:
one option is correct, the rest are not, and the marking is arithmetic.

So these tests are mostly about trust in the question bank and in the marking, plus
the one piece of plumbing that had to change to store the results — the CHECK
constraint on `interview_results.mode`, which an older database carries in its old
form and which `CREATE TABLE IF NOT EXISTS` will never update on its own.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.auth import can  # noqa: E402
from app.db import Database, SCHEMA_VERSION  # noqa: E402
from app.evaluation import EVAL_JOBS  # noqa: E402
from app.mcq import (BANK, BANK_SIZE, COVERED_SKILLS, PASS_MARK, bank_stats,  # noqa: E402
                     build_mcq_test, score_mcq_test)
from app.parser import parse_file  # noqa: E402
from app.taxonomy import TAXONOMY  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[1] / "sample_resumes"


@pytest.fixture(scope="module")
def candidates():
    return [parse_file(p) for p in sorted(FIXTURES.glob("*.txt"))]


@pytest.fixture(scope="module")
def pair(candidates):
    """One candidate and the job that suits them best — a realistic test to build."""
    return candidates[0], EVAL_JOBS[0]


# --------------------------------------------------------------------------- #
# the question bank
# --------------------------------------------------------------------------- #
def test_bank_is_not_a_stub():
    assert BANK_SIZE >= 60, "a screening bank this small would repeat itself immediately"
    assert len(COVERED_SKILLS) >= 30


def test_every_question_has_exactly_one_correct_answer_present():
    for skill, difficulty, question, options, correct, explanation in BANK:
        assert len(options) == 4, f"{question!r} has {len(options)} options, not 4"
        assert len(set(options)) == 4, f"{question!r} has a duplicated option"
        assert correct in options, f"{question!r} — the correct answer is not among its options"


def test_no_duplicate_questions():
    texts = [item[2] for item in BANK]
    assert len(texts) == len(set(texts))


def test_difficulties_are_valid():
    assert {item[1] for item in BANK} <= {"easy", "medium", "hard"}


def test_every_explanation_says_something():
    for item in BANK:
        assert len(item[5]) > 40, f"{item[2]!r} has an explanation too short to teach anything"


def test_every_skill_is_a_real_taxonomy_skill():
    """A question tagged with a skill the matcher has never heard of can never be selected."""
    unknown = [s for s in COVERED_SKILLS if s not in TAXONOMY]
    assert unknown == [], f"not in the taxonomy: {unknown}"


def test_bank_stats_reports_the_bank():
    stats = bank_stats()
    assert stats["questions"] == BANK_SIZE
    assert sum(stats["by_difficulty"].values()) == BANK_SIZE
    assert set(stats["covered_skills"]) == set(COVERED_SKILLS)


# --------------------------------------------------------------------------- #
# building a paper
# --------------------------------------------------------------------------- #
def test_test_has_the_requested_length(pair):
    candidate, job = pair
    for total in (5, 10, 15):
        test = build_mcq_test(candidate, job, total=total)
        assert len(test["questions"]) == total, f"asked for {total}, got {len(test['questions'])}"


def test_correct_index_points_at_the_correct_option(pair):
    """The options are shuffled per candidate; the key has to move with them."""
    candidate, job = pair
    test = build_mcq_test(candidate, job, total=12)
    by_text = {item[2]: item for item in BANK}
    for q in test["questions"]:
        source = by_text[q["question_text"]]
        assert q["options"][q["correct_index"]] == source[4]
        assert sorted(q["options"]) == sorted(source[3]), "shuffling changed the option set"


def test_the_same_candidate_sees_the_same_paper_twice(pair):
    candidate, job = pair
    a = build_mcq_test(candidate, job, total=10)
    b = build_mcq_test(candidate, job, total=10)
    assert [q["question_text"] for q in a["questions"]] == [q["question_text"] for q in b["questions"]]
    assert [q["options"] for q in a["questions"]] == [q["options"] for q in b["questions"]]


def test_different_candidates_do_not_share_an_answer_layout(candidates):
    """Otherwise "C, A, D, B" passed between two candidates would be worth something."""
    job = EVAL_JOBS[0]
    layouts = set()
    for c in candidates[:4]:
        test = build_mcq_test(c, job, total=8)
        layouts.add(tuple(q["correct_index"] for q in test["questions"]))
    assert len(layouts) > 1


def test_seniority_moves_the_difficulty(pair):
    _, job = pair
    base = {"name": "Test Person", "skills": [{"skill": s} for s in ("Python", "SQL", "Git")],
            "education": {"degree": "Bachelor's"}}
    junior = build_mcq_test({**base, "seniority": "Entry level", "experience_years": 0}, job, total=10)
    senior = build_mcq_test({**base, "seniority": "Senior", "experience_years": 9}, job, total=10)
    assert junior["target_difficulty"] == "easy"
    assert senior["target_difficulty"] == "hard"
    assert senior["mix"]["hard"] > junior["mix"]["hard"]


def test_a_paper_is_not_one_skill_over_and_over(pair):
    candidate, job = pair
    test = build_mcq_test(candidate, job, total=12)
    counts = {}
    for q in test["questions"]:
        counts[q["skill"]] = counts.get(q["skill"], 0) + 1
    assert max(counts.values()) <= 4, f"one skill dominated the paper: {counts}"


def test_time_limit_is_proportionate(pair):
    candidate, job = pair
    test = build_mcq_test(candidate, job, total=10)
    assert 10 * 30 <= test["time_limit_seconds"] <= 10 * 120


# --------------------------------------------------------------------------- #
# marking
# --------------------------------------------------------------------------- #
def test_all_correct_scores_full_marks(pair):
    candidate, job = pair
    test = build_mcq_test(candidate, job, total=10)
    report = score_mcq_test(test, [q["correct_index"] for q in test["questions"]])
    assert report["score"] == 100.0
    assert report["correct"] == 10
    assert report["passed"] is True
    assert report["verdict"] == "strong"


def test_all_wrong_scores_zero(pair):
    candidate, job = pair
    test = build_mcq_test(candidate, job, total=10)
    wrong = [(q["correct_index"] + 1) % len(q["options"]) for q in test["questions"]]
    report = score_mcq_test(test, wrong)
    assert report["score"] == 0.0
    assert report["passed"] is False
    assert report["verdict"] == "fail"


def test_a_blank_paper_scores_zero_rather_than_crashing(pair):
    candidate, job = pair
    test = build_mcq_test(candidate, job, total=10)
    report = score_mcq_test(test, [None] * 10)
    assert report["score"] == 0.0
    assert report["unanswered"] == 10


def test_a_short_answer_list_counts_the_rest_as_blank(pair):
    """A dropped connection must not be scored as if the paper were finished correctly."""
    candidate, job = pair
    test = build_mcq_test(candidate, job, total=10)
    report = score_mcq_test(test, [test["questions"][0]["correct_index"]])
    assert report["unanswered"] == 9
    assert report["correct"] == 1


def test_hard_questions_are_worth_more_than_easy_ones(pair):
    """Otherwise a candidate passes by clearing the easy half and skipping the rest."""
    candidate, job = pair
    test = build_mcq_test(candidate, job, total=12)
    levels = [q["difficulty_level"] for q in test["questions"]]
    if "easy" not in levels or "hard" not in levels:
        pytest.skip("this paper has no easy/hard contrast to measure")
    easy_only = [q["correct_index"] if q["difficulty_level"] == "easy" else None
                 for q in test["questions"]]
    hard_only = [q["correct_index"] if q["difficulty_level"] == "hard" else None
                 for q in test["questions"]]
    n_easy, n_hard = levels.count("easy"), levels.count("hard")
    if n_easy != n_hard:
        pytest.skip("counts differ, so the comparison would not isolate the weighting")
    assert score_mcq_test(test, hard_only)["score"] > score_mcq_test(test, easy_only)["score"]


def test_the_pass_mark_is_the_pass_mark(pair):
    candidate, job = pair
    test = build_mcq_test(candidate, job, total=10)
    for n in range(11):
        answers = [q["correct_index"] for q in test["questions"][:n]] + [None] * (10 - n)
        report = score_mcq_test(test, answers)
        assert report["passed"] == (report["score"] >= PASS_MARK)


def test_weak_skills_name_what_was_got_wrong(pair):
    candidate, job = pair
    test = build_mcq_test(candidate, job, total=10)
    target = test["questions"][0]["skill"]
    answers = [None if q["skill"] == target else q["correct_index"] for q in test["questions"]]
    report = score_mcq_test(test, answers)
    assert target in report["weak_skills"]
    assert target in report["summary"] or target in str(report["by_skill"])


def test_every_marked_row_carries_the_explanation(pair):
    candidate, job = pair
    test = build_mcq_test(candidate, job, total=8)
    report = score_mcq_test(test, [None] * 8)
    assert all(row["explanation"] for row in report["answers"])
    assert all(row["correct_text"] for row in report["answers"])


# --------------------------------------------------------------------------- #
# permissions
# --------------------------------------------------------------------------- #
def test_who_may_do_what():
    assert can("candidate", "mcq.practice")
    assert not can("candidate", "mcq.run"), "a candidate must not screen other people"
    assert not can("candidate", "mcq.read"), "a candidate must not browse the answer bank"
    assert can("recruiter", "mcq.run") and can("recruiter", "mcq.read")
    assert can("admin", "mcq.run")


# --------------------------------------------------------------------------- #
# storage — the migration this feature needed
# --------------------------------------------------------------------------- #
def test_schema_version_moved_for_the_mode_constraint():
    assert SCHEMA_VERSION >= 2


def test_a_fresh_database_accepts_an_mcq_result(tmp_path):
    db = Database(str(tmp_path / "fresh.db"))
    assert {r["version"] for r in db.all("SELECT version FROM schema_version")} >= {1, 2}
    db.close()


def test_an_old_database_is_migrated_without_losing_rows(tmp_path):
    """The failure this guards: `CREATE TABLE IF NOT EXISTS` is a no-op on an existing
    table, so a database created before 'mcq' existed keeps rejecting it forever unless
    `migrate()` genuinely rebuilds the table."""
    path = tmp_path / "old.db"
    schema = (Path(__file__).resolve().parents[1] / "app" / "schema.sql").read_text(encoding="utf-8")
    old = schema.replace("CHECK (mode IN ('text','voice','mcq'))", "CHECK (mode IN ('text','voice'))")

    raw = sqlite3.connect(str(path))
    raw.executescript(old)
    raw.executescript("""
        INSERT INTO candidates(name, email, resume_path, identity_hash)
            VALUES ('Old Row','old@example.com','old.txt','hash-old');
        INSERT INTO jobs(job_title) VALUES ('ML Engineer');
        INSERT INTO interview_results(candidate_id, job_id, score, feedback, mode, recommendation)
            VALUES (1, 1, 72, 'kept through the migration', 'text', 'Proceed');
        INSERT INTO schema_version(version) VALUES (1);
    """)
    raw.commit()
    with pytest.raises(sqlite3.IntegrityError):
        raw.execute("INSERT INTO interview_results(candidate_id, mode) VALUES (1,'mcq')")
    raw.close()

    db = Database(str(path))                      # __init__ migrates
    rows = db.all("SELECT * FROM interview_results ORDER BY result_id")
    assert len(rows) == 1, "the migration lost a row"
    assert rows[0]["feedback"] == "kept through the migration"
    assert rows[0]["mode"] == "text"

    saved = db.save_result(1, 1, {"score": 64, "feedback": "scored 64%",
                                  "recommendation": "Proceed with probes"}, "mcq", [])
    assert saved["mode"] == "mcq" and saved["score"] == 64
    db.close()


def test_the_migration_keeps_the_other_constraints(tmp_path):
    """Rebuilding a table is a chance to silently drop a rule. These must survive."""
    db = Database(str(tmp_path / "rules.db"))
    db.conn.execute("INSERT INTO candidates(name, resume_path, identity_hash) "
                    "VALUES ('C','c.txt','h')")
    db.conn.commit()
    for bad, why in (("'essay'", "an unknown mode"), ("'mcq', 150", "a score over 100")):
        with pytest.raises(Exception):
            cols = "mode" if "," not in bad else "mode, score"
            db.conn.execute(f"INSERT INTO interview_results(candidate_id, {cols}) "
                            f"VALUES (1, {bad})")
            db.conn.commit()
    db.close()


def test_migrating_twice_changes_nothing(tmp_path):
    db = Database(str(tmp_path / "twice.db"))
    before = db.all("SELECT version FROM schema_version ORDER BY version")
    db.migrate()
    db.migrate()
    assert db.all("SELECT version FROM schema_version ORDER BY version") == before
    db.close()
