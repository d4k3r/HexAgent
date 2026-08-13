from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from fixtures import (
    literal_terminal_fixture,
    normal_selfplay_fixture,
    normal_teacher_fixture,
    virtual_completion_fixture,
)
from hex_reconstruction.validation import ValidationError, validate_example
from hex_reconstruction.schema import read_jsonl, write_jsonl


class SchemaValidationTests(unittest.TestCase):
    def test_jsonl_round_trip_preserves_versioned_contract(self) -> None:
        fixtures = [normal_teacher_fixture(), normal_selfplay_fixture()]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "examples.jsonl"
            write_jsonl(path, fixtures)
            loaded = read_jsonl(path)
        self.assertEqual([item.to_dict() for item in loaded], [item.to_dict() for item in fixtures])

    def test_all_synthetic_fixtures_validate(self) -> None:
        for fixture in (
            normal_teacher_fixture(),
            normal_selfplay_fixture(),
            virtual_completion_fixture(),
            literal_terminal_fixture(),
        ):
            with self.subTest(fixture=fixture.game_id):
                validate_example(fixture)

    def test_rejects_policy_mass_on_illegal_action(self) -> None:
        fixture = normal_teacher_fixture()
        illegal = fixture.state.planes[0].index(1)
        fixture.policy.pi[illegal] = fixture.policy.pi[fixture.transition.chosen_action]
        fixture.policy.pi[fixture.transition.chosen_action] = 0.0
        with self.assertRaisesRegex(ValidationError, "illegal action"):
            validate_example(fixture)

    def test_rejects_non_normalized_policy(self) -> None:
        fixture = normal_teacher_fixture()
        fixture.policy.pi[fixture.transition.chosen_action] += 0.1
        with self.assertRaisesRegex(ValidationError, "sum to 1"):
            validate_example(fixture)

    def test_rejects_nullable_policy_with_positive_weight(self) -> None:
        fixture = virtual_completion_fixture()
        fixture.policy.weight = 1.0
        with self.assertRaisesRegex(ValidationError, "pi=null"):
            validate_example(fixture)

    def test_rejects_bad_shape_and_action(self) -> None:
        fixture = normal_selfplay_fixture()
        fixture.state.planes[0] = fixture.state.planes[0][:-1]
        with self.assertRaisesRegex(ValidationError, "121"):
            validate_example(fixture)

        fixture = normal_selfplay_fixture()
        fixture.transition.chosen_action = 121
        with self.assertRaisesRegex(ValidationError, "physical action"):
            validate_example(fixture)

    def test_rejects_terminal_inconsistency(self) -> None:
        fixture = deepcopy(virtual_completion_fixture())
        fixture.terminal.virtual_winner = None
        with self.assertRaisesRegex(ValidationError, "requires virtual_winner"):
            validate_example(fixture)

    def test_rejects_missing_provenance(self) -> None:
        fixture = normal_teacher_fixture()
        fixture.provenance.model_sha256 = ""
        with self.assertRaisesRegex(ValidationError, "model_sha256"):
            validate_example(fixture)


if __name__ == "__main__":
    unittest.main()
