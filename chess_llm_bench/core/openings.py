"""
Opening book for chess games.

Provides a small, unbiased collection of standard chess openings from the
Encyclopedia of Chess Openings (ECO) covering various opening systems.
The openings are represented as sequences of moves in UCI notation.
"""

from __future__ import annotations

import random
from typing import List, Dict, Tuple, Optional

# Standard openings from Encyclopedia of Chess Openings (ECO)
# Format: (ECO code, Opening name, [moves in UCI notation])
OPENINGS = [
    # A - Flank Openings
    ('A00', 'Polish Opening (Sokolsky)', ['b2b4']),
    ('A04', 'Reti Opening', ['g1f3', 'd7d5', 'g2g3']),
    ('A10', 'English Opening', ['c2c4', 'e7e5']),

    # B - Semi-Open Games (except French Defense)
    ('B00', 'Uncommon King\'s Pawn Opening', ['e2e4', 'b8c6']),  # Nimzowitsch Defense
    ('B02', 'Alekhine\'s Defense', ['e2e4', 'g8f6']),
    ('B20', 'Sicilian Defense', ['e2e4', 'c7c5']),

    # C - Open Games and French Defense
    ('C00', 'French Defense', ['e2e4', 'e7e6']),
    ('C20', 'King\'s Pawn Game', ['e2e4', 'e7e5']),
    ('C40', 'King\'s Knight Opening', ['e2e4', 'e7e5', 'g1f3']),
    ('C42', 'Petrov\'s Defense (Russian Game)', ['e2e4', 'e7e5', 'g1f3', 'g8f6']),
    ('C44', 'King\'s Pawn Game: Ponziani Opening', ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'c2c3']),
    ('C50', 'Italian Game', ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1c4']),
    ('C60', 'Ruy Lopez (Spanish Game)', ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5']),

    # D - Closed Games and Semi-Closed Games
    ('D00', 'Queen\'s Pawn Game', ['d2d4', 'd7d5']),
    ('D02', 'Queen\'s Pawn Game: London System', ['d2d4', 'd7d5', 'g1f3', 'g8f6', 'c1f4']),
    ('D06', 'Queen\'s Gambit', ['d2d4', 'd7d5', 'c2c4']),
    ('D30', 'Queen\'s Gambit Declined', ['d2d4', 'd7d5', 'c2c4', 'e7e6']),
    ('D50', 'Queen\'s Gambit Declined: Classical Variation', ['d2d4', 'd7d5', 'c2c4', 'e7e6', 'b1c3', 'g8f6', 'c1g5']),

    # E - Indian Defenses and King's Indian
    ('E00', 'Queen\'s Pawn Game: Neo-Indian (Catalan Opening)', ['d2d4', 'g8f6', 'c2c4', 'e7e6', 'g2g3']),
    ('E12', 'Queen\'s Indian Defense', ['d2d4', 'g8f6', 'c2c4', 'e7e6', 'g1f3', 'b7b6']),
    ('E60', 'King\'s Indian Defense', ['d2d4', 'g8f6', 'c2c4', 'g7g6'])
]


class OpeningBook:
    """
    A simple opening book that provides standard openings from ECO.
    """

    def __init__(self):
        """Initialize the opening book with standard ECO openings."""
        self.openings = OPENINGS

    def get_random_opening(self) -> Tuple[str, str, List[str]]:
        """
        Get a random opening from the book.

        Returns:
            A tuple of (ECO code, opening name, move list)
        """
        return random.choice(self.openings)

    def get_opening_by_eco(self, eco_code: str) -> Optional[Tuple[str, str, List[str]]]:
        """
        Get a specific opening by its ECO code.

        Args:
            eco_code: The ECO code to look up (e.g., "E00")

        Returns:
            The opening tuple if found, None otherwise
        """
        for opening in self.openings:
            if opening[0] == eco_code:
                return opening
        return None

    def get_all_openings(self) -> List[Tuple[str, str, List[str]]]:
        """
        Get all openings in the book.

        Returns:
            List of all openings
        """
        return self.openings

    def get_opening_categories(self) -> Dict[str, List[Tuple[str, str, List[str]]]]:
        """
        Group openings by their first letter category.

        Returns:
            Dictionary mapping category letter to list of openings
        """
        categories = {}
        for opening in self.openings:
            category = opening[0][0]  # First letter of ECO code
            if category not in categories:
                categories[category] = []
            categories[category].append(opening)
        return categories

    def get_balanced_pair(self) -> Tuple[Tuple[str, str, List[str]], Tuple[str, str, List[str]]]:
        """
        Get a balanced pair of openings for white and black.

        Returns:
            Two opening tuples, one for white and one for black oriented
        """
        # Get openings by category
        categories = self.get_opening_categories()

        # Try to pick from different opening families for balance
        if len(categories) >= 2:
            cat_keys = list(categories.keys())
            cat1, cat2 = random.sample(cat_keys, 2)
            opening1 = random.choice(categories[cat1])
            opening2 = random.choice(categories[cat2])
        else:
            # If we don't have enough categories, just pick two random openings
            openings = random.sample(self.openings, 2)
            opening1, opening2 = openings

        return opening1, opening2
