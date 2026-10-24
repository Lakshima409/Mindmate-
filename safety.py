"""
Crisis detection and safe response layer for MindMate.

This module provides a deterministic safety layer that runs BEFORE
ordinary intent-based response generation. It is designed to detect
expressions of self-harm, suicide risk, and mental health emergencies.

Key principles:
- Always err on the side of safety
- Never diagnose or make clinical claims
- Use verified crisis resources only
- Override intent prediction and confidence thresholds
- Do not generate harmful instructions
"""

import os
import re
from typing import Optional, Tuple

# Verified crisis resources (currently available regions)
CRISIS_RESOURCES = {
    "us_canada": {
        "hotline": "988",
        "description": "988 Suicide & Crisis Lifeline",
        "text_line": "Text HOME to 741741",
    },
    "uk": {
        "hotline": "111",
        "description": "NHS 111",
        "text_line": "Samaritans: 116 123",
    },
    "australia": {
        "hotline": "13 11 14",
        "description": "Lifeline Australia",
    },
    "new_zealand": {
        "hotline": "1737",
        "description": "Need to talk? (1737)",
    },
    "sri_lanka": {
        "hotline": "1926",
        "description": "National Mental Health Helpline",
        "emergency": {
            "police": "119",
            "ambulance": "1990",
        },
    },
}

# Default region for crisis resources.
# The documented deployment target is Sri Lanka, but this remains configurable
# through MINDMATE_CRISIS_REGION for other deployments. An invalid or empty
# value falls back to the documented default instead of silently using an
# unrelated region.
_CONFIGURED_CRISIS_REGION = os.environ.get("MINDMATE_CRISIS_REGION", "sri_lanka").strip().lower()
DEFAULT_REGION = (
    _CONFIGURED_CRISIS_REGION
    if _CONFIGURED_CRISIS_REGION in CRISIS_RESOURCES
    else "sri_lanka"
)

# Crisis detection patterns - verified, carefully reviewed keyword sets
# These patterns are designed to catch direct and indirect expressions of crisis
CRISIS_PATTERNS = [
    # Direct suicide ideation
    (r"\b(kill\s+my\s+self|kill\s+myself|end\s+my\s+life|end\s+it\s+all|suicide|ctb|cto|don't\s+want\s+to\s+live|done\s+with\s+life)\b", "direct_suicide"),
    # Suicide ideation
    (r"\b(want\s+to\s+die|dying|suicidal\s+thoughts)\b", "suicide_ideation"),
    # Self-harm
    (r"\b(hurt\s+myself|hurt\s+my\s+self|self\.harm|cut\s+myself|harm\s+my\s+body)\b", "self_harm"),
    # Hopelessness
    (r"\b(hopeless|nobody\s+cares|everyone\s+would\s+be\s+better\s+off|end\s+everything)\b", "hopelessness"),
    # Immediate emergency
    (r"\b(overdose|passing\s+out|emergency|help\s+me|someone\s+help)\b", "immediate_danger"),
]

# Non-crisis contexts that should NOT trigger (but we err on safety)
NON_CRISIS_CONTEXTS = [
    r"\b(research|paper|study|article|reading|learning)\b",
    r"\b(book|movie|show|game| fiction)\b",
]


def _normalizes(text: str) -> str:
    """Normalize text for pattern matching: lowercase, collapse whitespace, and
    correct a small set of common misspellings seen in crisis expressions."""
    normalized = " ".join(text.lower().split())
    replacements = {
        "myslef": "myself",
        "hopelesss": "hopeless",
        "wanna": "want to",
        "dont": "don't",
    }
    for wrong, right in replacements.items():
        normalized = re.sub(r"\b" + re.escape(wrong) + r"\b", right, normalized)
    return normalized


def _is_clearly_non_crisis_context(normalized: str) -> bool:
    """Return True only for clearly educational, media, quoted-third-party, or
    negated self-harm contexts.

    Suppression is intentionally conservative: a credible first-person crisis
    signal in the same message is never suppressed merely because an
    educational/hypothetical/quoted marker is also present.
    """
    direct_risk_markers = [
        "kill myself",
        "kill my self",
        "end my life",
        "end it all",
        "want to die",
        "dying",
        "suicidal thoughts",
        "hurt myself",
        "hurt my self",
        "self harm",
        "self.harm",
        "cut myself",
        "harm my body",
        "hopeless",
        "nobody cares",
        "everyone would be better off",
        "end everything",
        "overdose",
        "passing out",
        "emergency",
        "help me",
        "someone help",
        "done with life",
        "don't want to live",
        "do not want to live",
    ]

    negated_self_harm_markers = [
        "don't want to hurt myself",
        "do not want to hurt myself",
        "i don't want to hurt myself",
        "i do not want to hurt myself",
    ]
    if any(marker in normalized for marker in negated_self_harm_markers):
        without_negation = normalized
        for marker in negated_self_harm_markers:
            without_negation = without_negation.replace(marker, " ")
        if not any(marker in without_negation for marker in direct_risk_markers):
            return True

    quoted_third_party_markers = [
        "my friend said",
        "friend said",
    ]
    okay_markers = [
        "but i am okay",
        "but i'm okay",
        "but i am fine",
        "but i'm fine",
    ]
    if any(marker in normalized for marker in quoted_third_party_markers) and any(
        marker in normalized for marker in okay_markers
    ):
        unquoted = re.sub(r"'[^']*'|\"[^\"]*\"", " ", normalized)
        if not any(marker in unquoted for marker in direct_risk_markers):
            return True

    educational_markers = [
        "research paper",
        "reading about suicide",
        "suicide scene",
        "movie had",
        "hypothetically, why do people",
        "why do people attempt suicide",
        "why do people commit suicide",
    ]
    if any(marker in normalized for marker in educational_markers) and not any(
        marker in normalized for marker in direct_risk_markers
    ):
        return True

    return False


def resolve_crisis_region(region=None):
    """Resolve and validate the crisis-resource region.

    ``None`` uses the configured default. Invalid regions fall back to the
    documented default instead of returning an unrelated region's resources.
    """
    candidate = DEFAULT_REGION if region is None else region
    if not isinstance(candidate, str):
        candidate = DEFAULT_REGION
    candidate = candidate.strip().lower()
    if candidate not in CRISIS_RESOURCES:
        candidate = DEFAULT_REGION
    return candidate


def detect_crisis(sentence: str) -> Tuple[bool, Optional[str]]:
    """
    Detect if the user's message indicates a crisis.

    Returns:
        (is_crisis, category) where:
        - is_crisis: True if crisis patterns match
        - category: The crisis category, or None if no match

    Important: This function uses deterministic keyword/pattern matching
    and cannot assess the user's actual level of risk. It should always
    be accompanied by appropriate help resources and a compassionate response.
    """
    normalized = _normalizes(sentence)

    # Narrow, evidence-based suppression for clearly non-crisis contexts.
    # This runs before pattern matching but never suppresses a direct
    # first-person imminent-risk signal outside those contexts.
    if _is_clearly_non_crisis_context(normalized):
        return False, None

    # Check each crisis pattern
    for pattern, category in CRISIS_PATTERNS:
        if re.search(pattern, normalized):
            return True, category

    return False, None


def is_safe_to_proceed(user_msg: str) -> bool:
    """
    Check if it's safe to proceed with normal response generation.

    Crisis patterns are checked FIRST. If any crisis pattern matches,
    we do NOT proceed with normal response generation, regardless of
    non-crisis context. We always err on the side of safety.

    Non-crisis contexts are only relevant after confirming no crisis
    patterns match, to reduce false positives for academic or literary
    discussions that happen to use similar language.
    """
    normalized = _normalizes(user_msg)

    # Keep safety-first behavior consistent with detect_crisis, including the
    # narrow context handling above.
    is_crisis, _ = detect_crisis(user_msg)
    return not is_crisis


def get_crisis_response(category: str, region: str = None) -> str:
    """
    Get a compassionate crisis response with verified resources.

    Args:
        category: The crisis category from detect_crisis()
        region: The region key for crisis resources (default: "us_canada")

    Returns:
        An empathetic response with appropriate help resources
    """
    resources = CRISIS_RESOURCES[resolve_crisis_region(region)]

    # Base compassionate response
    base = (
        "I'm concerned about what you've shared. You are not alone, "
        "and there is support available.\n"
    )

    # Category-specific resources and responses
    if category == "direct_suicide":
        return (
            base
            + resources["description"]
            + ": "
            + resources["hotline"]
            + "\n"
            + "If you are in immediate danger, please call emergency services.\n"
        )

    elif category == "suicide_ideation":
        return (
            base
            + "I'm here to listen and support you. "
            + resources["description"]
            + " is available at "
            + resources["hotline"]
            + ".\n"
            + "You don't have to face this alone. There are people who want to help."
        )

    elif category == "self_harm":
        return (
            base
            + "I'm sorry you're going through such deep pain. "
            + resources["description"]
            + " is available at "
            + resources["hotline"]
            + ".\n"
            + "Please reach out to someone who can provide immediate support. "
            + "You deserve to get help."
        )

    elif category == "hopelessness":
        return (
            base
            + "I can hear how much pain you're in right now. "
            + resources["description"]
            + " is available at "
            + resources["hotline"]
            + ".\n"
            + "Things can and do get better, even when it feels impossible. "
            + "Please reach out for support."
        )

    elif category == "immediate_danger":
        emergency_police = resources.get("emergency", {}).get("police", "")
        emergency_ambulance = resources.get("emergency", {}).get("ambulance", "")
        return (
            base
            + "If you are in immediate danger, please call emergency services right away.\n"
            + ("Police: " + emergency_police + "\n" if emergency_police else "")
            + ("Ambulance: " + emergency_ambulance + " for medical emergency\n" if emergency_ambulance else "")
            + resources["description"]
            + ": "
            + resources["hotline"]
            + "\n"
            + "If life is in immediate danger, call emergency services or go to the nearest emergency room. "
            + "You are not alone."
        )

    else:
        # Default response for unknown categories
        return (
            base
            + resources["description"]
            + ": "
            + resources["hotline"]
            + "\n"
            + "Please consider reaching out to a trusted person or professional for support."
        )
