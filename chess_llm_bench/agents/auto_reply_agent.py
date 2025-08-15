"""
Auto Reply Agent

This module provides an AutoReplyAgent that can automatically respond to
specific types of messages or game situations. It's useful for handling
standard protocol responses, acknowledgments, and automated communication
between chess agents in the AG2 framework.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Pattern, Union

import chess
from autogen import ConversableAgent, LLMConfig

from .base import ChessAgent, ChessAgentConfig
from ..core.models import BotSpec, LiveState

logger = logging.getLogger(__name__)


class AutoReplyRule:
    """
    Represents a single auto-reply rule.

    A rule consists of a trigger condition and a response action.
    """

    def __init__(
        self,
        name: str,
        trigger: Union[str, Pattern, Callable],
        response: Union[str, Callable],
        priority: int = 0,
        enabled: bool = True
    ):
        """
        Initialize an auto-reply rule.

        Args:
            name: Human-readable name for the rule
            trigger: Pattern or function to match incoming messages
            response: Response string or function to generate response
            priority: Priority level (higher = checked first)
            enabled: Whether this rule is active
        """
        self.name = name
        self.trigger = trigger
        self.response = response
        self.priority = priority
        self.enabled = enabled
        self.usage_count = 0

    def matches(self, message: str, context: Dict[str, Any] = None) -> bool:
        """Check if this rule matches the given message."""
        if not self.enabled:
            return False

        try:
            if isinstance(self.trigger, str):
                return self.trigger.lower() in message.lower()
            elif isinstance(self.trigger, Pattern):
                return bool(self.trigger.search(message))
            elif callable(self.trigger):
                return self.trigger(message, context or {})
            else:
                return False
        except Exception as e:
            logger.warning(f"Error checking rule {self.name}: {e}")
            return False

    def generate_response(self, message: str, context: Dict[str, Any] = None) -> str:
        """Generate response for the matched message."""
        self.usage_count += 1

        try:
            if isinstance(self.response, str):
                return self.response
            elif callable(self.response):
                return self.response(message, context or {})
            else:
                return f"Auto-reply from rule: {self.name}"
        except Exception as e:
            logger.error(f"Error generating response for rule {self.name}: {e}")
            return "Auto-reply error occurred"


class AutoReplyAgent(ConversableAgent):
    """
    Agent that automatically responds to specific message patterns.

    This agent can handle standard chess protocol responses, acknowledgments,
    and automated communication scenarios. It's useful for:
    - Protocol compliance in chess games
    - Standard responses to common queries
    - Error handling and status updates
    - Coordination between other agents
    """

    def __init__(
        self,
        name: str = "AutoReplyBot",
        rules: Optional[List[AutoReplyRule]] = None,
        default_response: Optional[str] = None,
        enable_learning: bool = False,
        **kwargs
    ):
        """
        Initialize the auto-reply agent.

        Args:
            name: Name of the auto-reply agent
            rules: List of auto-reply rules
            default_response: Default response when no rules match
            enable_learning: Whether to learn from interactions (future feature)
            **kwargs: Additional AG2 ConversableAgent arguments
        """
        # Initialize AG2 ConversableAgent
        super().__init__(
            name=name,
            system_message=self._create_system_message(),
            llm_config=None,  # Auto-reply doesn't need LLM
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,  # Allow multiple auto-replies
            **kwargs
        )

        # Auto-reply specific configuration
        self.rules: List[AutoReplyRule] = rules or []
        self.default_response = default_response or "Auto-reply: Message received"
        self.enable_learning = enable_learning

        # Statistics
        self.stats = {
            "messages_processed": 0,
            "auto_replies_sent": 0,
            "rules_matched": 0,
            "default_responses": 0,
        }

        # Add default chess-related rules
        self._add_default_rules()

        # Sort rules by priority
        self._sort_rules()

        logger.info(f"Initialized AutoReplyAgent: {name} with {len(self.rules)} rules")

    def _create_system_message(self) -> str:
        """Create system message for auto-reply agent."""
        return f"""You are {self.name}, an automated response agent for chess games.

Your responsibilities include:
1. Automatically responding to specific message patterns
2. Handling standard chess protocol communications
3. Providing acknowledgments and status updates
4. Facilitating communication between chess agents
5. Managing routine responses to reduce manual intervention

You operate based on predefined rules and patterns, providing quick and
consistent responses to common scenarios in chess games and tournaments.

You help maintain smooth communication flow in multi-agent chess environments.
"""

    def _add_default_rules(self) -> None:
        """Add default auto-reply rules for common chess scenarios."""

        # Game status acknowledgments
        self.add_rule(AutoReplyRule(
            name="game_start_ack",
            trigger=re.compile(r"game\s+(starting|started|begin)", re.IGNORECASE),
            response="Acknowledged: Game started. Ready to play.",
            priority=10
        ))

        self.add_rule(AutoReplyRule(
            name="game_end_ack",
            trigger=re.compile(r"game\s+(over|ended|finished)", re.IGNORECASE),
            response="Acknowledged: Game finished. Thank you for playing.",
            priority=10
        ))

        # Move acknowledgments
        self.add_rule(AutoReplyRule(
            name="move_received",
            trigger=re.compile(r"move\s+played|opponent\s+played", re.IGNORECASE),
            response="Move received and processed.",
            priority=5
        ))

        # Status requests
        self.add_rule(AutoReplyRule(
            name="status_request",
            trigger=re.compile(r"(status|how\s+are\s+you|ready)", re.IGNORECASE),
            response=lambda msg, ctx: self._generate_status_response(ctx),
            priority=8
        ))

        # Error handling
        self.add_rule(AutoReplyRule(
            name="error_ack",
            trigger=re.compile(r"error|failed|problem", re.IGNORECASE),
            response="Error acknowledged. Attempting to recover.",
            priority=7
        ))

        # Time management
        self.add_rule(AutoReplyRule(
            name="time_warning",
            trigger=re.compile(r"time\s+(running\s+out|low|warning)", re.IGNORECASE),
            response="Time warning received. Will prioritize move selection.",
            priority=6
        ))

        # Protocol responses
        self.add_rule(AutoReplyRule(
            name="ping_response",
            trigger="ping",
            response="pong",
            priority=9
        ))

        # Greeting responses
        self.add_rule(AutoReplyRule(
            name="greeting",
            trigger=re.compile(r"(hello|hi|greetings)", re.IGNORECASE),
            response=f"Hello! I'm {self.name}, ready for chess communication.",
            priority=3
        ))

    def add_rule(self, rule: AutoReplyRule) -> None:
        """Add a new auto-reply rule."""
        self.rules.append(rule)
        self._sort_rules()
        logger.debug(f"Added auto-reply rule: {rule.name}")

    def remove_rule(self, rule_name: str) -> bool:
        """Remove an auto-reply rule by name."""
        for i, rule in enumerate(self.rules):
            if rule.name == rule_name:
                del self.rules[i]
                logger.debug(f"Removed auto-reply rule: {rule_name}")
                return True
        return False

    def enable_rule(self, rule_name: str) -> bool:
        """Enable a rule by name."""
        for rule in self.rules:
            if rule.name == rule_name:
                rule.enabled = True
                logger.debug(f"Enabled auto-reply rule: {rule_name}")
                return True
        return False

    def disable_rule(self, rule_name: str) -> bool:
        """Disable a rule by name."""
        for rule in self.rules:
            if rule.name == rule_name:
                rule.enabled = False
                logger.debug(f"Disabled auto-reply rule: {rule_name}")
                return True
        return False

    def _sort_rules(self) -> None:
        """Sort rules by priority (highest first)."""
        self.rules.sort(key=lambda rule: rule.priority, reverse=True)

    def generate_reply(
        self,
        messages: List[Dict[str, Any]],
        sender: Optional[ConversableAgent] = None,
        config: Optional[Dict] = None
    ) -> Union[str, Dict, None]:
        """Generate auto-reply based on the incoming message."""
        if not messages:
            return None

        # Get the last message
        last_message = messages[-1]
        message_content = last_message.get("content", "")

        if not message_content:
            return None

        self.stats["messages_processed"] += 1

        # Create context for rule evaluation
        context = {
            "sender": sender.name if sender else "unknown",
            "message_count": len(messages),
            "agent_stats": self.stats.copy(),
        }

        # Try to match rules
        for rule in self.rules:
            if rule.matches(message_content, context):
                response = rule.generate_response(message_content, context)
                self.stats["auto_replies_sent"] += 1
                self.stats["rules_matched"] += 1

                logger.debug(f"Auto-reply rule '{rule.name}' matched: {response}")
                return response

        # No rules matched, use default response if configured
        if self.default_response:
            self.stats["auto_replies_sent"] += 1
            self.stats["default_responses"] += 1
            return self.default_response

        # No auto-reply
        return None

    def _generate_status_response(self, context: Dict[str, Any]) -> str:
        """Generate a status response with current agent information."""
        active_rules = len([r for r in self.rules if r.enabled])
        return (
            f"Status: Online and ready. "
            f"Active rules: {active_rules}, "
            f"Messages processed: {self.stats['messages_processed']}, "
            f"Auto-replies sent: {self.stats['auto_replies_sent']}"
        )

    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get statistics about rule usage."""
        rule_stats = {}
        for rule in self.rules:
            rule_stats[rule.name] = {
                "enabled": rule.enabled,
                "priority": rule.priority,
                "usage_count": rule.usage_count
            }

        return {
            "total_rules": len(self.rules),
            "active_rules": len([r for r in self.rules if r.enabled]),
            "rule_details": rule_stats,
            "agent_stats": self.stats.copy()
        }

    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self.stats = {
            "messages_processed": 0,
            "auto_replies_sent": 0,
            "rules_matched": 0,
            "default_responses": 0,
        }

        for rule in self.rules:
            rule.usage_count = 0

        logger.info("Auto-reply statistics reset")

    def export_rules(self) -> List[Dict[str, Any]]:
        """Export rules configuration for backup/sharing."""
        exported_rules = []
        for rule in self.rules:
            rule_data = {
                "name": rule.name,
                "priority": rule.priority,
                "enabled": rule.enabled,
                "usage_count": rule.usage_count,
            }

            # Handle different trigger types
            if isinstance(rule.trigger, str):
                rule_data["trigger_type"] = "string"
                rule_data["trigger_value"] = rule.trigger
            elif isinstance(rule.trigger, Pattern):
                rule_data["trigger_type"] = "regex"
                rule_data["trigger_value"] = rule.trigger.pattern
            else:
                rule_data["trigger_type"] = "function"
                rule_data["trigger_value"] = str(rule.trigger)

            # Handle response types
            if isinstance(rule.response, str):
                rule_data["response_type"] = "string"
                rule_data["response_value"] = rule.response
            else:
                rule_data["response_type"] = "function"
                rule_data["response_value"] = str(rule.response)

            exported_rules.append(rule_data)

        return exported_rules

    def __str__(self) -> str:
        """String representation of auto-reply agent."""
        active_rules = len([r for r in self.rules if r.enabled])
        return f"AutoReplyAgent({self.name}, {active_rules} active rules)"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"AutoReplyAgent(name='{self.name}', "
            f"total_rules={len(self.rules)}, "
            f"active_rules={len([r for r in self.rules if r.enabled])}, "
            f"messages_processed={self.stats['messages_processed']})"
        )


def create_auto_reply_agent(
    name: str = "AutoReplyBot",
    include_chess_rules: bool = True,
    **kwargs
) -> AutoReplyAgent:
    """
    Convenience function to create an auto-reply agent.

    Args:
        name: Name for the auto-reply agent
        include_chess_rules: Whether to include default chess rules
        **kwargs: Additional agent parameters

    Returns:
        Configured AutoReplyAgent
    """
    agent = AutoReplyAgent(name=name, **kwargs)

    if not include_chess_rules:
        # Remove default rules if not wanted
        agent.rules.clear()

    return agent


def create_protocol_agent(name: str = "ProtocolBot") -> AutoReplyAgent:
    """
    Create an auto-reply agent specifically for chess protocol handling.

    Args:
        name: Name for the protocol agent

    Returns:
        AutoReplyAgent configured for protocol handling
    """
    agent = AutoReplyAgent(
        name=name,
        default_response=None  # Only respond to specific protocol messages
    )

    # Add protocol-specific rules
    agent.add_rule(AutoReplyRule(
        name="fen_request",
        trigger=re.compile(r"fen|position", re.IGNORECASE),
        response="FEN position request acknowledged.",
        priority=10
    ))

    agent.add_rule(AutoReplyRule(
        name="move_notation",
        trigger=re.compile(r"[a-h][1-8]|[KQRBN][a-h]?[1-8]", re.IGNORECASE),
        response="Chess move notation detected.",
        priority=8
    ))

    return agent
