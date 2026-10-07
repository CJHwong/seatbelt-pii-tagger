import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("agentwork_prepare", Path(__file__).parent / "prepare.py")
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


def test_the_overlap_gate_reads_both_transcript_sets(tmp_path):
    (tmp_path / "test").mkdir()
    (tmp_path / "sample.jsonl").write_text(json.dumps({"text": "dev window"}) + "\n")
    (tmp_path / "test" / "sample.jsonl").write_text(json.dumps({"text": "test window"}) + "\n")
    assert prepare.transcript_texts(tmp_path) == ["dev window", "test window"]


def test_a_machine_without_transcripts_adds_nothing(tmp_path):
    assert prepare.transcript_texts(tmp_path) == []
