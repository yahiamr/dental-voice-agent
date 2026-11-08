# Intent check results

Cases: `evals/intent_cases.yaml`. Commit: `a49e633-dirty`. Run on 2026-11-08 18:57.

## fixed
Time per case: median 0.000 s, p90 0.000 s, max 0.000 s. Errors: 0 of 120.

**emergency** (24 true, 30 false)
| Layer | Caught | Missed | False flags | Missed ids | False flag ids |
|---|---|---|---|---|---|
| fixed | 7 | 17 | 6 | e01 e03 e04 e05 e06 e07 e08 e09 e10 e11 e12 e13 e14 e15 e17 e22 e23 | n04 n06 n07 n09 n22 n23 |

**claims_change** (12 true, 20 false)
| Layer | Caught | Missed | False flags | Missed ids | False flag ids |
|---|---|---|---|---|---|
| fixed | 10 | 2 | 0 | r04 r11 | - |

**medical_advice** (7 true, 25 false)
| Layer | Caught | Missed | False flags | Missed ids | False flag ids |
|---|---|---|---|---|---|
| fixed | 1 | 6 | 1 | r22 r23 r24 r25 r26 r27 | r32 |

**yes** (16 plain yes, 18 not)
| Layer | Right | Wrong | Wrong ids |
|---|---|---|---|
| fixed | 33 | 1 | y15 |

Combined, where only the fixed rule can write:
- Plain yes: written: 15 (y01 y02 y03 y04 y05 y06 y07 y08 y09 y10 y11 y12 y13 y14 y16); not a yes: 1 (y15)
- Not a yes: not a yes: 18 (x01 x02 x03 x04 x05 x06 x07 x08 x09 x10 x11 x12 x13 x14 x15 x16 x17 x18)

## jev:typesafe/jev-1.14
Time per case: median 0.242 s, p90 0.295 s, max 1.041 s. Errors: 2 of 120.
First error: r18: TimeoutError: The read operation timed out

**emergency** (24 true, 30 false)
| Layer | Caught | Missed | False flags | Missed ids | False flag ids |
|---|---|---|---|---|---|
| fixed | 7 | 17 | 6 | e01 e03 e04 e05 e06 e07 e08 e09 e10 e11 e12 e13 e14 e15 e17 e22 e23 | n04 n06 n07 n09 n22 n23 |
| model | 24 | 0 | 0 | - | - |
| product | 24 | 0 | 0 | - | - |

**claims_change** (12 true, 20 false)
| Layer | Caught | Missed | False flags | Missed ids | False flag ids |
|---|---|---|---|---|---|
| fixed | 10 | 2 | 0 | r04 r11 | - |
| model | 12 | 0 | 1 | - | r16 |
| product | 12 | 0 | 1 | - | r16 |

**medical_advice** (7 true, 25 false)
| Layer | Caught | Missed | False flags | Missed ids | False flag ids |
|---|---|---|---|---|---|
| fixed | 1 | 6 | 1 | r22 r23 r24 r25 r26 r27 | r32 |
| model | 7 | 0 | 1 | - | r32 |
| product | 7 | 0 | 1 | - | r32 |

**yes** (16 plain yes, 18 not)
| Layer | Right | Wrong | Wrong ids |
|---|---|---|---|
| fixed | 33 | 1 | y15 |
| model | 31 | 3 | y07 y14 y15 |

Combined, where only the fixed rule can write:
- Plain yes: written: 15 (y01 y02 y03 y04 y05 y06 y07 y08 y09 y10 y11 y12 y13 y14 y16); not a yes: 1 (y15)
- Not a yes: not a yes: 18 (x01 x02 x03 x04 x05 x06 x07 x08 x09 x10 x11 x12 x13 x14 x15 x16 x17 x18)

## anthropic:claude-haiku-5-5
Time per case: median 0.717 s, p90 0.922 s, max 1.427 s. Errors: 0 of 120.

**emergency** (24 true, 30 false)
| Layer | Caught | Missed | False flags | Missed ids | False flag ids |
|---|---|---|---|---|---|
| fixed | 7 | 17 | 6 | e01 e03 e04 e05 e06 e07 e08 e09 e10 e11 e12 e13 e14 e15 e17 e22 e23 | n04 n06 n07 n09 n22 n23 |
| model | 24 | 0 | 0 | - | - |
| product | 24 | 0 | 6 | - | n04 n06 n07 n09 n22 n23 |

**claims_change** (12 true, 20 false)
| Layer | Caught | Missed | False flags | Missed ids | False flag ids |
|---|---|---|---|---|---|
| fixed | 10 | 2 | 0 | r04 r11 | - |
| model | 12 | 0 | 0 | - | - |
| product | 12 | 0 | 0 | - | - |

**medical_advice** (7 true, 25 false)
| Layer | Caught | Missed | False flags | Missed ids | False flag ids |
|---|---|---|---|---|---|
| fixed | 1 | 6 | 1 | r22 r23 r24 r25 r26 r27 | r32 |
| model | 7 | 0 | 1 | - | r32 |
| product | 7 | 0 | 1 | - | r32 |

**yes** (16 plain yes, 18 not)
| Layer | Right | Wrong | Wrong ids |
|---|---|---|---|
| fixed | 33 | 1 | y15 |
| model | 33 | 1 | y14 |

Combined, where only the fixed rule can write:
- Plain yes: written: 15 (y01 y02 y03 y04 y05 y06 y07 y08 y09 y10 y11 y12 y13 y14 y16); asked again: 1 (y15)
- Not a yes: not a yes: 18 (x01 x02 x03 x04 x05 x06 x07 x08 x09 x10 x11 x12 x13 x14 x15 x16 x17 x18)
