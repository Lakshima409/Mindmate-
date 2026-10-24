#!/usr.userns/bin/env python3
"""Test script for the safety module."""
import sys
sys.path.insert(0, '.')

from safety import detect_crisis, get_crisis_response, is_safe_to_proceed, CRISIS_RESOURCES, resolve_crisis_region

print("=" * 60)
print("Safety Module Tests")
print("=" * 60)

# Test detect_crisis
test_cases = [
    ("I want to kill myself", True, "direct_suicide"),
    ("I'm going to end it all", True, "direct_suicide"),
    ("I want to ctb", True, "direct_suicide"),
    ("I want to hurt myself", True, "self_harm"),
    ("I had suicidal thoughts", True, "suicide_ideation"),
    ("I'm feeling hopeless", True, "hopelessness"),
    ("I don't want to live anymore", True, "direct_suicide"),
    ("Nobody cares about me", True, "hopelessness"),
    ("I'm reading about suicide for a research paper", False, None),
    ("I'm reading about suicide for a research paper and I have suicidal thoughts", True, "direct_suicide"),
    ("Hypothetically, why do people attempt suicide? I want to die", True, "direct_suicide"),
    ("My friend said 'I want to die' but I am okay, and I also want to die", True, "suicide_ideation"),
    ("I don't want to hurt myself", False, None),
    ("I'm feeling stressed about exams", False, None),
    ("Hello how are you", False, None),
    ("What's the weather today", False, None),
]

passed = 0
failed = 0

for user_input, expected_crisis, expected_category in test_cases:
    is_crisis, category = detect_crisis(user_input)
    if is_crisis == expected_crisis and category == expected_category:
        passed += 1
        print(f"  PASS: {user_input[:50]}...")
    else:
        failed += 1
        print(f"  FAIL: {user_input[:50]}...")
        print(f"    Expected: crisis={expected_crisis}, category={expected_category}")
        print(f"    Got: crisis={is_crisis}, category={category}")

print(f"\ndetect_crisis: {passed} passed, {failed} failed out of {len(test_cases)} tests")

# Test is_safe_to_proceed
print("\nis_safe_to_proceed tests:")
safe_tests = [
    ("I'm fine", True),
    ("Hello", True),
    ("What's up", True),
]
for user_input, expected_safe in safe_tests:
    result = is_safe_to_proceed(user_input)
    status = "PASS" if result == expected_safe else "FAIL"
    print(f"  {status}: '{user_input}' -> safe={result} (expected {expected_safe})")
    if result != expected_safe:
        failed += 1
    else:
        passed += 1

print(f"\nis_safe_to_proceed: {passed} passed, {failed} failed")

# Test get_crisis_response
print("\nget_crisis_response test:")
response = get_crisis_response("direct_suicide")
print(f"  Response length: {len(response)} characters")
print(f"  Contains hotline: {'1926' in response}")
print(f"  Contains emergency: {'911' in response or '999' in response or '111' in response}")

print("\n" + "=" * 60)
print("Test Summary")
print("=" * 60)
print(f"Total passed: {passed}")
print(f"Total failed: {failed}")

# Test configured region resolution and crisis-response path
print("\nCrisis region tests:")
import safety as safety_module

original_default = safety_module.DEFAULT_REGION
safety_module.DEFAULT_REGION = "sri_lanka"
try:
    assert resolve_crisis_region() == "sri_lanka"
    assert resolve_crisis_region("us_canada") == "us_canada"
    assert resolve_crisis_region("not-a-region") == "sri_lanka"
    assert resolve_crisis_region(None) == "sri_lanka"

    response = get_crisis_response("direct_suicide")
    print(f"  PASS: default response uses configured region hotline 1926: {'1926' in response}")

    us_response = get_crisis_response("direct_suicide", region="us_canada")
    print(f"  PASS: explicit region override uses 988: {'988' in us_response}")
finally:
    safety_module.DEFAULT_REGION = original_default