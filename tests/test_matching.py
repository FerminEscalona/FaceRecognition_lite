import copy
import json

import numpy as np
import pytest

from face_lite.evaluation import calibrate, summarize
from face_lite.matching import Calibration, Gallery, decide


def rows(person="p", version="v"):
    vectors = ([1, .01, 0], [1, .02, 0], [1, .03, 0])
    return [{"template_id": f"{person}-{i}", "person_id": person, "model_version": version,
             "embedding": (np.array(v) / np.linalg.norm(v)).tolist()} for i, v in enumerate(vectors)]


def policy(threshold=.8, margin=.1):
    return Calibration("v", threshold, margin, "validation-only", .3)


def test_json_roundtrip_and_top_two_scoring():
    source = rows()
    g = Gallery(json.loads(json.dumps(source)), "v", 3)
    source[0]["embedding"][0] = 0  # Gallery owns a copy.
    expected = np.mean(sorted([1/np.sqrt(1+x*x) for x in (.01, .02, .03)])[-2:])
    assert g.scores([1, 0, 0])[0][1] == pytest.approx(expected)


@pytest.mark.parametrize("mutation", [
    lambda r: r.pop(),
    lambda r: r[0].update(model_version="other"),
    lambda r: r[0].update(embedding=[1, 0]),
    lambda r: r[0].update(embedding=[float("nan"), 0, 0]),
    lambda r: r[0].update(embedding=[2, 0, 0]),
    lambda r: r[0].update(embedding=[0, 0, 0]),
    lambda r: r[0].update(embedding=["1", "0", "0"]),
    lambda r: r[1].update(template_id=r[0]["template_id"]),
    lambda r: r[1].update(embedding=r[0]["embedding"]),
])
def test_reject_incompatible_or_incomplete_rows(mutation):
    data = rows()
    mutation(data)
    with pytest.raises(ValueError):
        Gallery(data, "v", 3)


def test_empty_unknown_ambiguous_and_ties():
    p = policy()
    assert decide([], p)["person_id"] is None
    assert decide([("p", .7)], p)["status"] == "unknown"
    assert decide([("p", .9), ("q", .85)], p)["status"] == "ambiguous"
    assert decide([("p", .9), ("q", .9)], policy(margin=0))["status"] == "ambiguous"
    assert decide([("p", .9), ("q", .5)], p)["person_id"] == "p"
    assert decide([("p", .9)], p)["person_id"] == "p"


def record(truth, scores, status="ok"):
    return {"truth": truth, "subject_id": truth or "outsider", "scores": scores,
            "status": status, "elapsed_ms": 1}


def test_joint_calibration_avoids_wrong_identity_and_keeps_failures_in_denominator():
    validation = [record("p", [("p", .9), ("q", .4)]),
                  record("q", [("q", .85), ("p", .5)]),
                  record(None, [("p", .95), ("q", .94)]),
                  record("p", [], "no_face")]
    p = calibrate(validation, "v", "digest", .3)
    stats = summarize(validation, p)
    assert stats["incorrect_assignments"] == 0
    assert stats["known_identification_rate"] == pytest.approx(2/3)
    assert stats["status_counts"]["no_face"] == 1
    assert stats["known_identification_wilson95"][0] < 2/3
    # Test results may be bad; they must never change the frozen calibration.
    frozen = copy.deepcopy(p.to_dict())
    test = [record(None, [("p", 1), ("q", 0)])]
    assert summarize(test, p)["incorrect_assignments"] == 1
    assert p.to_dict() == frozen


def test_reject_all_is_valid_but_does_not_pass_quality():
    validation = [record("p", [("q", .99)]), record(None, [("q", .99)])]
    p = calibrate(validation, "v", "digest", .3)
    assert summarize(validation, p)["known_identification_rate"] == 0
    assert summarize(validation, p)["incorrect_assignments"] == 0


def test_consistency_checks_all_pairs():
    r = rows()
    r[2]["embedding"] = [-1, 0, 0]
    with pytest.raises(ValueError, match="consistency"):
        Gallery(r, "v", 3).check_consistency(.3)

