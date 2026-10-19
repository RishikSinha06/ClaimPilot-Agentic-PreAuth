Base model: Mock LLM (simulated, offline). 12 labelled cases × 12 faults; only runs where the fault changed the LLM's answer are counted.

| Fault | What it does | Runs | Correct | Held for review | Wrong decision out | Letter blocked | Caught by |
|---|---|---|---|---|---|---|---|
| malformed_json | Returns broken JSON on every attempt | 12 | 12 | 0 | 0 | — | schema validation → rule fallback (12) |
| hallucinated_condition | Adds a chronic condition the notes never mention | 12 | 12 | 0 | 0 | — | grounding (12) |
| negated_condition | Reports a condition the notes explicitly negate | 3 | 3 | 0 | 0 | — | grounding (3) |
| wrong_duration | Says every chronic condition started 6 months ago | 2 | 2 | 0 | 0 | — | Critic: disagreement does not change the outcome (1), grounding (1) |
| dropped_conditions | Leaves out every chronic condition | 2 | 2 | 0 | 0 | — | Critic: disagreement does not change the outcome (1), Critic: extractors disagree → human (1) |
| wrong_procedure | Swaps the procedure (dental surgery <-> appendectomy) | 12 | 3 | 9 | 0 | — | Critic: extractors disagree → human (10), Critic: denial kept only on grounds both extractors support (2) |
| flipped_accident | Flips the accident flag | 12 | 3 | 9 | 0 | — | Critic: extractors disagree → human (10), Critic: denial kept only on grounds both extractors support (2) |
| flipped_cosmetic | Flips the cosmetic-intent flag | 12 | 6 | 6 | 0 | — | Critic: extractors disagree → human (7), Critic: disagreement does not change the outcome (3), Critic: denial kept only on grounds both extractors support (2) |
| flipped_functional | Flips the functional-need flag | 12 | 11 | 1 | 0 | — | Critic: disagreement does not change the outcome (11), Critic: extractors disagree → human (1) |
| letter_changed_amount | Changes (or adds) a ₹ amount in the letter | 12 | 12 | 0 | 0 | 12/12 | faithfulness check (12) |
| letter_dropped_reason | Drops one reason from the letter | 12 | 12 | 0 | 0 | 12/12 | faithfulness check (12) |
| letter_flipped_decision | Rewrites the letter with the opposite decision | 12 | 12 | 0 | 0 | 12/12 | faithfulness check (12) |

**115 faulty runs: 0 wrong decision(s) and 0 corrupted letter(s) got out.**
