# Retrieval results

Document: `data/docs/patient-handbook.md`. Embedding model: `BAAI/bge-small-en-v1.5`. Commit: `55c4556-dirty`. Run on 2026-11-08 20:43.

31 labelled questions in the caller's own words, plus 5 the handbook does not answer. A hit means an answering piece is among the 3 the model would see.

| Search | Hits at 3 | Hit rate | MRR at 5 | direct | reworded | lookalike | odd_place |
| --- | --- | --- | --- | --- | --- | --- | --- |
| keyword | 28 of 31 | 90% | 0.85 | 9 of 10 | 7 of 9 | 7 of 7 | 5 of 5 |
| vector | 30 of 31 | 97% | 0.90 | 10 of 10 | 8 of 9 | 7 of 7 | 5 of 5 |
| hybrid | 30 of 31 | 97% | 0.91 | 10 of 10 | 8 of 9 | 7 of 7 | 5 of 5 |

**Misses** (no answering piece in the top 3):

- keyword: d04 "What are your opening hours?", w03 "What happens if I just don't turn up?", w04 "My mouth still feels frozen after the injection"
- vector: w03 "What happens if I just don't turn up?"
- hybrid: w03 "What happens if I just don't turn up?"

**Unanswerable questions**, with hybrid search's top piece:

- u01 "Do you do Botox?": `frequently-asked-questions/do-you-see-patients-who-are-nervous-about-the-dentist`
- u02 "Can I bring my dog?": `frequently-asked-questions/can-i-bring-my-child-to-my-appointment`
- u03 "Is there a vending machine?": `frequently-asked-questions/do-you-have-a-toilet-for-patients`
- u04 "Do you sell electric toothbrushes?": `about-the-clinic/what-we-do-and-what-we-do-not-do`
- u05 "Can I get a COVID test at the clinic?": `finding-us`
