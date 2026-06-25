from backend.token_alignment import align_token_sequences


def test_alignment_preserves_order_and_marks_changes():
    current = ["The", "robot", "is", "touching", "an", "object"]
    reference = ["The", "robot", "is", "holding", "a", "green", "cube"]
    rows = align_token_sequences(current, reference)
    pairs = [(row["current_index"], row["reference_index"]) for row in rows
             if row["current_index"] is not None and row["reference_index"] is not None]
    assert pairs == sorted(pairs)
    assert any(row["operation"] == "replace" for row in rows)
    assert any(row["operation"] in {"insert", "delete"} for row in rows)


def test_identical_sequences_are_all_matches():
    rows = align_token_sequences(["The", "cube"], ["The", "cube"])
    assert [row["operation"] for row in rows] == ["match", "match"]


if __name__ == "__main__":
    test_alignment_preserves_order_and_marks_changes()
    test_identical_sequences_are_all_matches()
    print("token alignment tests passed")
