"""
Unit tests for bot parsing and move validation functionality.

Tests the core parsing logic for bot specifications and LLM move extraction
to ensure robust handling of various input formats and edge cases.
"""

import unittest
import re
from chess_llm_bench.llm.client import parse_bot_spec, MOVE_REGEX
from chess_llm_bench.core.models import BotSpec


class BotParsingTests(unittest.TestCase):
    """Test bot specification parsing."""

    def test_parse_bots_single_random(self):
        """Test parsing single random bot spec."""
        bots = parse_bot_spec("random::test_bot")
        self.assertEqual(len(bots), 1)
        self.assertEqual(bots[0].provider, "random")
        self.assertEqual(bots[0].model, "")
        self.assertEqual(bots[0].name, "test_bot")

    def test_parse_bots_single_openai(self):
        """Test parsing single OpenAI bot spec."""
        bots = parse_bot_spec("openai:gpt-4o-mini:gpt4o")
        self.assertEqual(len(bots), 1)
        self.assertEqual(bots[0].provider, "openai")
        self.assertEqual(bots[0].model, "gpt-4o-mini")
        self.assertEqual(bots[0].name, "gpt4o")

    def test_parse_bots_multiple(self):
        """Test parsing multiple bot specifications."""
        bots = parse_bot_spec("random::foo,openai:gpt-4o-mini:bar")
        self.assertEqual(len(bots), 2)

        self.assertEqual(bots[0].provider, "random")
        self.assertEqual(bots[0].model, "")
        self.assertEqual(bots[0].name, "foo")

        self.assertEqual(bots[1].provider, "openai")
        self.assertEqual(bots[1].model, "gpt-4o-mini")
        self.assertEqual(bots[1].name, "bar")

    def test_parse_bots_name_with_colons(self):
        """Test parsing bot name that contains colons."""
        bots = parse_bot_spec("openai:gpt-4:my:complex:name")
        self.assertEqual(len(bots), 1)
        self.assertEqual(bots[0].provider, "openai")
        self.assertEqual(bots[0].model, "gpt-4")
        self.assertEqual(bots[0].name, "my:complex:name")

    def test_parse_bots_two_parts(self):
        """Test parsing bot spec with only two parts."""
        bots = parse_bot_spec("openai:gpt-4o-mini")
        self.assertEqual(len(bots), 1)
        self.assertEqual(bots[0].provider, "openai")
        self.assertEqual(bots[0].model, "gpt-4o-mini")
        self.assertEqual(bots[0].name, "gpt-4o-mini")

    def test_parse_bots_single_part(self):
        """Test parsing bot spec with only provider."""
        bots = parse_bot_spec("random")
        self.assertEqual(len(bots), 1)
        self.assertEqual(bots[0].provider, "random")
        self.assertEqual(bots[0].model, "")
        self.assertEqual(bots[0].name, "random")

    def test_parse_bots_invalid_provider(self):
        """Test parsing with invalid provider raises error."""
        with self.assertRaises(ValueError) as context:
            parse_bot_spec("invalid_provider::test")
        self.assertIn("Unsupported provider", str(context.exception))

    def test_parse_bots_empty_string(self):
        """Test parsing empty string returns empty list."""
        bots = parse_bot_spec("")
        self.assertEqual(len(bots), 0)

    def test_parse_bots_whitespace_only(self):
        """Test parsing whitespace-only string returns empty list."""
        bots = parse_bot_spec("   ")
        self.assertEqual(len(bots), 0)

    def test_parse_bots_with_spaces(self):
        """Test parsing bot specs with extra spaces."""
        bots = parse_bot_spec(" random::bot1 , openai:gpt-4:bot2 ")
        self.assertEqual(len(bots), 2)
        self.assertEqual(bots[0].name, "bot1")
        self.assertEqual(bots[1].name, "bot2")

    def test_parse_bots_case_insensitive_provider(self):
        """Test that provider names are case-insensitive."""
        bots = parse_bot_spec("RANDOM::test,OpenAI:gpt-4:test2")
        self.assertEqual(len(bots), 2)
        self.assertEqual(bots[0].provider, "random")
        self.assertEqual(bots[1].provider, "openai")

    def test_parse_bots_anthropic(self):
        """Test parsing Anthropic bot specification."""
        bots = parse_bot_spec("anthropic:claude-3-haiku:claude")
        self.assertEqual(len(bots), 1)
        self.assertEqual(bots[0].provider, "anthropic")
        self.assertEqual(bots[0].model, "claude-3-haiku")
        self.assertEqual(bots[0].name, "claude")

    def test_parse_bots_empty_model(self):
        """Test parsing bot spec with empty model field."""
        bots = parse_bot_spec("random::my_bot")
        self.assertEqual(len(bots), 1)
        self.assertEqual(bots[0].provider, "random")
        self.assertEqual(bots[0].model, "")
        self.assertEqual(bots[0].name, "my_bot")


class MoveRegexTests(unittest.TestCase):
    """Test chess move regex patterns."""

    def test_move_regex_simple_pawn_move(self):
        """Test simple pawn move extraction."""
        match = MOVE_REGEX.search("The best move is e2e4")
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), "e2e4")

    def test_move_regex_knight_move(self):
        """Test knight move extraction."""
        match = MOVE_REGEX.search("I suggest g1f3")
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), "g1f3")

    def test_move_regex_queen_promotion(self):
        """Test queen promotion move extraction."""
        match = MOVE_REGEX.search("Promote with a7a8q")
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), "a7a8q")

    def test_move_regex_knight_promotion(self):
        """Test knight promotion move extraction."""
        match = MOVE_REGEX.search("Best is h7h8n for knight promotion")
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), "h7h8n")

    def test_move_regex_rook_promotion(self):
        """Test rook promotion move extraction."""
        match = MOVE_REGEX.search("go with b7b8r")
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), "b7b8r")

    def test_move_regex_bishop_promotion(self):
        """Test bishop promotion move extraction."""
        match = MOVE_REGEX.search("choose c7c8b")
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), "c7c8b")

    def test_move_regex_case_insensitive(self):
        """Test that regex is case insensitive."""
        match = MOVE_REGEX.search("Play E2E4")
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), "E2E4")

    def test_move_regex_multiple_moves(self):
        """Test extracting first move when multiple present."""
        match = MOVE_REGEX.search("Consider e2e4 or d2d4")
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), "e2e4")

    def test_move_regex_no_match_invalid_format(self):
        """Test no match for invalid move format."""
        match = MOVE_REGEX.search("invalid move format")
        self.assertIsNone(match)

    def test_move_regex_no_match_partial(self):
        """Test no match for partial move."""
        match = MOVE_REGEX.search("e2e")
        self.assertIsNone(match)

    def test_move_regex_no_match_wrong_squares(self):
        """Test no match for invalid square names."""
        match = MOVE_REGEX.search("z9z8")
        self.assertIsNone(match)

    def test_move_regex_boundaries(self):
        """Test that regex respects word boundaries."""
        match = MOVE_REGEX.search("some1e2e4text")
        self.assertIsNone(match)

        match = MOVE_REGEX.search("some e2e4 text")
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), "e2e4")

    def test_move_regex_with_punctuation(self):
        """Test move extraction with surrounding punctuation."""
        match = MOVE_REGEX.search("Best move: e2e4!")
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), "e2e4")

    def test_move_regex_castling_not_matched(self):
        """Test that castling notation is not matched by UCI regex."""
        match = MOVE_REGEX.search("Castle with O-O")
        self.assertIsNone(match)

        match = MOVE_REGEX.search("Long castle O-O-O")
        self.assertIsNone(match)


class BotSpecTests(unittest.TestCase):
    """Test BotSpec data model."""

    def test_bot_spec_creation(self):
        """Test creating a valid BotSpec."""
        spec = BotSpec(provider="openai", model="gpt-4", name="test_bot")
        self.assertEqual(spec.provider, "openai")
        self.assertEqual(spec.model, "gpt-4")
        self.assertEqual(spec.name, "test_bot")

    def test_bot_spec_case_normalization(self):
        """Test that provider name is normalized to lowercase."""
        spec = BotSpec(provider="OpenAI", model="gpt-4", name="test")
        self.assertEqual(spec.provider, "openai")

    def test_bot_spec_empty_provider_error(self):
        """Test that empty provider raises ValueError."""
        with self.assertRaises(ValueError) as context:
            BotSpec(provider="", model="gpt-4", name="test")
        self.assertIn("Provider cannot be empty", str(context.exception))

    def test_bot_spec_empty_name_error(self):
        """Test that empty name raises ValueError."""
        with self.assertRaises(ValueError) as context:
            BotSpec(provider="openai", model="gpt-4", name="")
        self.assertIn("Bot name cannot be empty", str(context.exception))

    def test_bot_spec_string_representation(self):
        """Test string representation of BotSpec."""
        spec = BotSpec(provider="openai", model="gpt-4", name="test_bot")
        self.assertEqual(str(spec), "test_bot (openai:gpt-4)")

    def test_bot_spec_string_representation_no_model(self):
        """Test string representation with empty model."""
        spec = BotSpec(provider="random", model="", name="random_bot")
        self.assertEqual(str(spec), "random_bot (random)")


if __name__ == "__main__":
    unittest.main()
