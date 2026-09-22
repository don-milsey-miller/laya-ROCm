"""Shared question/state fixtures that exercise every branch of sequence construction."""

LONG_TEXT = "The customer was charged twice for the same invoice and wants a refund today. " * 80

QUESTIONS = {
    "route": {"type": "choice", "instructions": "Which team should handle this?",
              "criteria": {"billing": "payments and invoices", "technical": "bugs", "sales": "pricing"}},
    "bare": {"type": "choice", "instructions": "Pick one", "criteria": ["yes", "no", "maybe"]},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["not urgent", "somewhat", "urgent", "critical"]},
    "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
    "noul_crit": {"type": "noul", "instructions": "Is it spam?",
                  "criteria": {"true": "unsolicited advertising", "false": ""}},
    "structured": {"type": "choice", "instructions": {"task": "classify", "fields": ["a", "b"]},
                   "criteria": {"x": {"desc": "rubric", "weight": 2}, "y": 0, "z": None}},
    # 12 long options: forces the option budget branch in build_sequence
    "crowded": {"type": "choice", "instructions": "Which of many intents? " * 30,
                "criteria": {("intent_%d" % i): ("a long description of intent number %d " % i) * 12
                             for i in range(12)}},
    "masky": {"type": "choice", "instructions": "Contains [MASK] token text",
              "criteria": {"a": "option with [MASK] inside", "b": "plain"}},
}

STATES = [
    "short text",
    {"text": "This works surprisingly well."},
    [{"role": "user", "content": "I was charged twice"}, {"role": "assistant", "content": "Sorry!"}],
    {"ticket": {"subject": "Payout failing", "body": LONG_TEXT}},
    "unicode: café — 東京 — मदद करें [MASK]",
    "",
]
