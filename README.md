# Cross-Lingual Payload Drift Detection

| | |
| --- | --- |
| Final rank | not ranked |
| Domain | Fine-Tuning |
| Difficulty | Medium |
| Scoring | ↑ Higher is better |
| Compute | CPU |
| Challenge status | Accepted / closed |
| Solutions submitted | 1 |
| Last submission | 2026-08-13 |

## Problem statement

### Overview

Cross-Lingual Payload Drift Detection is a hard CPU fine-tuning challenge about repairing disagreement across several textual views of the same underlying command.

Each sample contains:

- One English anchor request.
- Four mixed-language echoes.
- Several plausible payload disagreements spread across those echoes.
- One hidden minimal repair ledger.

The four echoes are not independent questions.

They are competing observations of one command.

Most wording is already correct.

A small number of payload fragments have drifted.

The model must identify those local disagreements, determine what the shared command actually requires, and generate the ordered set of repairs needed to bring every echo back into agreement.

A conceptual sample may look like:

Anchor:

set an alarm for 7 tomorrow morning

Echo E01:

kal subah 8 baje alarm laga do

Echo E02:

kal subah 7 baje alarm laga do

Echo E03:

parso subah 7 baje alarm laga do

Echo E04:

kal subah 7 baje alarm set karo

The hidden repair ledger could conceptually require:

- Repair the time payload in E01.
- Repair the day payload in E03.
- Leave the other payloads untouched.

The participant does not output a corrected paragraph.

The participant outputs a compact patch sequence.

A conceptual target may look like:

<PATCH> E01 L018 <OLD> 8 baje </OLD> <NEW> 7 baje </NEW> </PATCH>

<PATCH> E03 L041 <OLD> parso </OLD> <NEW> kal </NEW> </PATCH>

The `L###` symbols are opaque repair lanes.

Their meanings are stable across the released challenge, but their numeric IDs carry no semantic information.

A model must therefore solve five coupled problems:

- Detect which echoes contain semantic drift.
- Localize the incorrect payload span.
- Infer the intended replacement from the anchor and the other echoes.
- Infer the correct opaque repair lane.
- Emit all repairs in the required ledger order.

This is not ordinary translation.

It is not grammatical error correction.

It is not intent classification.

It is not semantic-tree generation.

It is a multi-view discrepancy-repair task where the target is the **minimal patch program** that reconciles several partially trustworthy observations.

### Why the Task Uses Multiple Views

A single noisy utterance can be ambiguous.

Four echoes provide redundancy.

For any recoverable semantic payload, at least one public view retains useful evidence about the intended value.

However, the model is not told which echo is trustworthy for which payload.

Different echoes can be correct in different places.

A useful model should combine evidence across the entire sample rather than choose one echo as globally authoritative.

The English anchor supplies an additional semantic reference.

It can help resolve disagreements when the mixed-language echoes disagree with one another.

The challenge therefore resembles a small textual sensor-fusion problem:

- The anchor provides one view.
- Each echo provides another view.
- Each view can preserve or distort different payloads.
- The output records only the repairs needed to restore consistency.

### The Repair Ledger

Every target is a sequence of one or more patch records.

Each patch uses the form:

<PATCH> E## L### <OLD> old text </OLD> <NEW> new text </NEW> </PATCH>

For example:

<PATCH> E02 L007 <OLD> 9 pm </OLD> <NEW> 8 pm </NEW> </PATCH>

A patch contains four important pieces of information:

- Which echo must be changed.
- Which learned repair lane is affected.
- Which visible span is wrong.
- Which replacement span restores the shared command.

The output contains only actual repairs.

Correct text is not repeated.

This makes the target sparse.

A model must distinguish:

- Evidence that confirms the shared command.
- Evidence that contradicts the shared command.
- Text that is grammatical framing.
- Text that represents a mutable payload.

### Echo IDs

Every sample contains exactly four echo IDs:

- E01
- E02
- E03
- E04

Echo IDs are local to the sample.

They identify the four public observations.

They do not have a stable quality ranking.

E01 is not systematically cleaner than E04.

The model should not treat echo number as a reliability feature.

The repair ledger is ordered first by echo ID:

E01 before E02 before E03 before E04.

Within one echo, repairs are ordered by the left-to-right position of the incorrect span.

This gives every sample one deterministic target ordering.

### Repair Lanes

Every patch also contains one opaque lane ID.

Lane IDs use the form:

L001

L002

L003

...

A repair lane identifies the recurring semantic role of the payload being repaired.

The public label is intentionally opaque.

For example, the challenge does not expose labels such as:

- destination
- date
- duration
- recipient
- reminder text
- artist
- event name

Instead, those recurring repair channels are represented by arbitrary `L###` symbols.

The mapping is globally stable.

If `L018` is associated with the same kind of payload in two training examples, it remains `L018`.

The numeric ordering is random.

Participants should not assume:

- Similar lane numbers have similar meanings.
- Low lane numbers are common.
- High lane numbers are rare.
- Consecutive lanes belong to the same task family.
- Lane numbers encode source position.

The lane system forces the model to learn how local textual disagreements map onto recurring repair behavior.

Pretrained language knowledge can help understand the utterances.

It cannot directly reveal the lane mapping.

### Minimality

The gold ledger contains only necessary changes.

Suppose E02 already contains the correct payload.

The model should not emit a patch that replaces that payload with an identical string.

Suppose two echoes disagree, but one matches the shared command supported by the anchor and the remaining views.

Only the inconsistent echo should be patched.

A prediction that edits already-correct text is over-repair.

A prediction that misses an inconsistent payload is under-repair.

Both are penalized.

The challenge therefore rewards **minimal semantic intervention**.

### Payload Drift

Drift occurs only inside short payload-bearing spans.

Examples may involve:

- Times.
- Dates.
- Durations.
- Locations.
- Names.
- Recipients.
- Message content.
- Reminder content.
- Music references.
- Event descriptions.
- Numeric quantities.
- Navigation endpoints.
- Weather locations.
- Other short task-bearing phrases.

Drift is designed to remain locally plausible.

A wrong payload should still look like text that could naturally occur in that position.

For example:

- One valid city can replace another city.
- One valid time can replace another time.
- One valid contact name can replace another name.
- One valid duration can replace another duration.

The task is not to detect nonsense.

The task is to detect semantic inconsistency across views.

### Code-Switched Echoes

The echo text mixes English and Romanized Indic phrasing in Latin script.

A single echo may contain:

- English task vocabulary inside Indic syntax.
- Indic function words around English names.
- Mixed date and time expressions.
- English named entities inside mostly non-English wording.
- Informal Romanization.
- Spelling variation.
- Lower-case text.
- Missing punctuation.
- Short imperative constructions.

The amount of switching varies.

The same semantic payload can surface differently across the anchor and echoes.

For example, an anchor may use:

tomorrow morning

while an echo may use:

kal subah

The model therefore cannot solve the task by exact substring agreement alone.

### Public Input

Each sample exposes five textual views:

- One English `anchor`.
- Four mixed-language `echoes`.

The four echo IDs are always:

- E01
- E02
- E03
- E04

Training samples additionally expose `repair_target`.

Test samples omit `repair_target`.

The exact field types, row counts, and full JSON examples are provided in the Dataset section.

### Training Target

`repair_target` is one string containing the complete ordered repair ledger.

A conceptual target is:

<PATCH> E01 L018 <OLD> 8 baje </OLD> <NEW> 7 baje </NEW> </PATCH> <PATCH> E03 L041 <OLD> parso </OLD> <NEW> kal </NEW> </PATCH>

Every patch has exactly this structural order:

- `<PATCH>`
- Echo ID.
- Lane ID.
- `<OLD>`
- Incorrect visible span.
- `</OLD>`
- `<NEW>`
- Correct replacement text.
- `</NEW>`
- `</PATCH>`

Ordinary whitespace between structural items is not semantically meaningful.

Whitespace inside payload spans is canonicalized during evaluation.

### Repair Count

Every sample requires between one and four patches.

The released build is deliberately concentrated around two- and three-patch cases rather than single-edit cases.

Across the full challenge:

- Approximately 20% of samples contain one repair.
- Approximately 40% contain two repairs.
- Approximately 30% contain three repairs.
- Approximately 10% contain four repairs.
- The median repair count is 2.
- The mean repair count is approximately 2.3.

The exact counts are fixed in the released files.

Participants should not assume a constant repair count.

The model must decide when to stop emitting patches.

A complete prediction must contain every required patch exactly once.

Duplicate repair keys are structurally redundant and are penalized.

### What Must Be Learned

The task contains several layers of learnable structure.

### Cross-View Agreement

The model must determine which payload value is supported by the sample as a whole.

This can require:

- Comparing the anchor with all four echoes.
- Recognizing paraphrases.
- Recognizing cross-language equivalents.
- Ignoring grammatical wording differences.
- Distinguishing semantic payloads from surrounding syntax.

### Drift Localization

The model must identify the wrong span in the affected echo.

This requires exact enough generation to reproduce the visible incorrect text.

### Replacement Recovery

The model must generate the intended replacement span.

The replacement may be:

- Copied from another echo.
- Closely paraphrased by another echo.
- Supported by the English anchor.
- Recoverable only after combining several views.

### Lane Induction

The model must infer which opaque `L###` lane corresponds to the repaired payload role.

### Sparse Sequencing

The model must output only the edits that matter and serialize them in deterministic order.

### Challenge Type

This is a **Fine-Tuning** challenge.

Generic pretrained sequence-to-sequence models are allowed.

The intended solution is a learned conditional generator that consumes the anchor and four echoes and emits the repair ledger.

Suitable model families include:

- Compact T5-style encoder-decoder models.
- Small BART-style encoder-decoder models.
- Compact multilingual text-to-text models.
- Byte-level encoder-decoder models.
- Small pretrained models with task-specific structural tokens.
- Pointer-augmented encoder-decoder systems.
- Compact models with learned cross-view pooling.
- CPU-friendly ensembles of eligible learned models.

The challenge does not require a large language model.

The intended regime is compact fine-tuning under CPU constraints.

### Machine-Learning Requirement

The primary predictive system must be learned from released challenge examples.

The following are not eligible as the main solution:

- Hand-written keyword dispatch.
- Manually authored lane dictionaries.
- Rule-only repair systems.
- Regular-expression-only drift detection.
- TF-IDF retrieval.
- Nearest-neighbor target replay.
- Template matching as the main predictor.
- Manual language-specific translation tables.
- Hard-coded payload correction rules.
- Exact memorization of training ledgers followed by lookup.

Deterministic structural post-processing is allowed.

Examples include:

- Blocking malformed patch tags.
- Preventing duplicate patch keys.
- Canonicalizing whitespace.
- Enforcing legal echo IDs.
- Enforcing legal lane IDs discovered from training.
- Sorting already-predicted patches into canonical order.

Those procedures may constrain syntax.

They must not determine the semantic repairs themselves.

### Compute

The execution environment provides:

- 10 CPU cores.
- 62.5 GiB RAM.
- No GPU.

The challenge is designed for compact fine-tuning.

Useful CPU strategies include:

- Small pretrained encoder-decoder checkpoints.
- Compact source formatting.
- Short target length caps.
- Dynamic padding.
- Length bucketing.
- Cached tokenization.
- Gradient accumulation.
- Frozen lower layers during early experiments.
- Parameter-efficient fine-tuning where useful.
- Small beam widths.
- Constrained decoding.
- Early stopping on a participant-created local validation split.
- One or a few compact models instead of very large ensembles.

The public text is short enough that participants do not need long-context architectures.

### Dataset

The prepared competition dataset contains **16,800 samples in total**.

The split is fixed as:

- **14,400 training samples**
- **2,400 test samples**
- **No official validation file**
- **4 mixed-language echoes per sample**
- **60 opaque repair lanes**
- **1 to 4 gold repairs per sample**

Across the complete prepared dataset there are:

- **16,800 English anchors**
- **67,200 mixed-language echoes**
- **Approximately 38,600 gold patch records**
- **60 distinct `L###` repair-lane symbols**
- **A median of 2 repairs per sample**
- **A mean of approximately 2.3 repairs per sample**

The repair-count distribution is approximately:

- **20%** of samples require 1 repair.
- **40%** require 2 repairs.
- **30%** require 3 repairs.
- **10%** require 4 repairs.

Every repair lane required by the test set is represented in the training set.

The numeric lane IDs are scrambled and have no semantic ordering.

The public package contains exactly three files:

- `train.jsonl`
- `test.jsonl`
- `sample_submission.csv`

The private evaluator uses hidden answer data that is not part of the public package.

### train.jsonl

`train.jsonl` contains exactly **14,400 JSON objects**, one object per line.

Each object contains four fields:

- `sample_id`
- `anchor`
- `echoes`
- `repair_target`

##### sample_id

Type: string.

Opaque identifier used only for submission alignment.

Example:

`PPL_0A17C4D2`

The identifier must not be used as a predictive feature.

##### anchor

Type: string.

An English realization of the command shared by the sample.

Example:

`Add a new weekly reminder for Sunday Brunch at 9 : 30 am`

##### echoes

Type: list of four objects.

Each echo object contains:

- `echo_id`: string
- `text`: string

The four IDs are always:

- E01
- E02
- E03
- E04

Each `text` field contains one mixed-language Latin-script realization of the same command.

Different echoes may contain different payload drifts.

##### repair_target

Type: string.

The complete ordered gold repair ledger.

It contains every patch required to reconcile the four public echoes with the command expressed by the sample.

One training row is:

{"sample_id":"PPL_0A17C4D2","anchor":"Add a new weekly reminder for Sunday Brunch at 9 : 30 am","echoes":[{"echo_id":"E01","text":"9 : 30 am ko Sunday Brunch ke liye ek naya weekly reminder add karen"},{"echo_id":"E02","text":"8 : 30 am ko Sunday Brunch ke liye ek naya weekly reminder add karen"},{"echo_id":"E03","text":"9 : 30 am ko Saturday Brunch ke liye ek naya weekly reminder add karen"},{"echo_id":"E04","text":"9 : 30 am ko Sunday Brunch ke liye ek naya monthly reminder add karen"}],"repair_target":"<PATCH> E02 L017 <OLD> 8 : 30 am </OLD> <NEW> 9 : 30 am </NEW> </PATCH> <PATCH> E03 L044 <OLD> Saturday Brunch </OLD> <NEW> Sunday Brunch </NEW> </PATCH> <PATCH> E04 L052 <OLD> monthly </OLD> <NEW> weekly </NEW> </PATCH>"}

For this row:

- E01 requires no repair.
- E02 contains one time drift.
- E03 contains one reminder-description drift.
- E04 contains one recurrence drift.
- The target contains exactly three patches.
- The lane IDs are opaque public symbols rather than descriptive field names.

### test.jsonl

`test.jsonl` contains exactly **2,400 JSON objects**, one object per line.

Each object contains three fields:

- `sample_id`
- `anchor`
- `echoes`

The structure of `sample_id`, `anchor`, and `echoes` is identical to the training file.

The only omitted field is:

`repair_target`

One test row has the following structure:

{"sample_id":"PPL_7D42B19E","anchor":"Send Alex a message saying I will arrive at six","echoes":[{"echo_id":"E01","text":"Alex ko message bhejo ki main six baje pahunchunga"},{"echo_id":"E02","text":"Alex ko message bhejo ki main seven baje pahunchunga"},{"echo_id":"E03","text":"Alex ko bolo ki main six baje pahunchunga"},{"echo_id":"E04","text":"Sam ko message bhejo ki main six baje pahunchunga"}]}

For a test row, participants must infer:

- Which echoes require repair.
- Which lane applies to each repair.
- The visible OLD span.
- The intended NEW span.
- The complete patch sequence.

The public test file does not expose:

- Gold patch keys.
- Gold OLD spans.
- Gold NEW spans.
- Gold repair counts.
- Canonical repaired echoes.
- Lane meanings.
- Construction metadata.
- Source partition metadata.

### sample_submission.csv

`sample_submission.csv` contains exactly **2,400 rows**, one for every `sample_id` in `test.jsonl`.

It contains exactly two columns:

- `sample_id`
- `prediction`

`sample_id` is copied directly from the test set.

`prediction` is initially blank.

The file begins in this form:

sample_id,prediction

PPL_7D42B19E,

PPL_18F0A6C3,

PPL_A931D50B,

Participants should preserve:

- The same 2,400 sample IDs.
- The same column order.
- One row per test sample.
- No duplicate rows.
- No additional columns.

The completed `prediction` cell must contain the entire generated repair ledger for that sample.

### Lane Inventory

The prepared training data contains **60 distinct repair lanes**.

They are serialized as:

- L001
- L002
- L003
- ...
- L060

The IDs are globally stable across training and test.

There is no separate lane dictionary.

Participants infer lane behavior from examples in `train.jsonl`.

Every lane appearing in hidden test answers has at least one training occurrence.

Lane numbers do not encode:

- Frequency.
- Semantic category.
- Source position.
- Echo identity.
- Difficulty.
- Repair count.

### Length and Sparsity Characteristics

The source side of each sample is compact:

- One anchor.
- Four echoes.
- Five total text views.

The target side is sparse.

Only incorrect payloads are written into `repair_target`.

Most words visible in the source do not appear in the target.

Typical target variation comes from:

- Number of required repairs.
- OLD-span length.
- NEW-span length.
- Number of distinct repair lanes.
- Whether several echoes drift on related payloads.
- Whether the replacement is copied from another echo or inferred from the anchor.

The combination of short inputs and sparse targets is intended to keep fine-tuning feasible on CPU while preserving a difficult multi-view reasoning problem.

### Drift Constellations

The challenge does not use a plain random split.

A sample can contain several repair keys such as:

E01 / L018

E03 / L041

E04 / L006

The set and co-occurrence pattern of repaired lanes across echoes forms a **drift constellation**.

The official evaluation partition is designed so that the exact repair constellation of a test sample does not appear as a training target.

In addition, test examples contain at least one lane co-repair relationship that is not reproduced as the same local repair pairing in released training examples.

Individual lanes remain learnable.

The challenge therefore tests whether a model can recombine familiar repair behaviors in unfamiliar multi-view disagreement patterns.

This is different from memorizing a complete ledger template.

For local validation, participants should avoid a purely random row split when possible.

A more realistic split groups examples by repair constellation or rare lane co-repair patterns.

### Submission Format

The submission contains one row per test sample.

The columns are exactly:

- `sample_id`
- `prediction`

The `prediction` field contains the complete generated repair ledger.

Example:

sample_id,prediction

PPL_example,"<PATCH> E01 L018 <OLD> 8 baje </OLD> <NEW> 7 baje </NEW> </PATCH> <PATCH> E03 L041 <OLD> parso </OLD> <NEW> kal </NEW> </PATCH>"

The submission must contain:

- Every expected `sample_id`.
- No missing IDs.
- No extra IDs.
- No duplicate IDs.
- Exactly the two published columns in the published order.

Use `sample_submission.csv` exactly.

### Patch Parsing

The evaluator parses every prediction into a sequence of patch records.

A patch is valid only when it contains:

- One legal echo ID.
- One syntactically valid lane ID.
- One non-empty OLD span.
- One non-empty NEW span.
- Properly nested structural tags.

A prediction can contain several valid patches.

If one patch is malformed, that patch is not credited as a valid repair record.

Catastrophic submission-schema errors still receive the minimum overall score.

### Patch Key

Every repair has a patch key:

echo_id + lane_id

For example:

E03 / L041

The gold ledger never contains the same patch key twice.

A prediction that repeats the same key creates redundant repairs.

For structural matching, only the first predicted occurrence of a duplicate key is used.

Duplicate records still reduce sequence and replay quality.

### Canonical Span Text

For OLD and NEW spans, the evaluator applies:

- HTML entity decoding.
- Unicode NFKC normalization.
- Ordinary whitespace normalization.

Case is preserved for exact-span matching.

A separate token-level component compares spans case-insensitively.

This allows partial credit for near-correct lexical recovery.

### Evaluation

Submissions are scored with the **Parallax Ledger Score** from 0.01 to 100.

Higher is better.

The score evaluates six published components:

- Patch Key F1.
- OLD Span F1.
- NEW Span F1.
- Ledger Order LCS.
- Replay Fidelity.
- Exact Ledger.

There are no hidden metric components.

A perfect prediction receives exactly 100.

### Patch Key F1

For each sample, compare the set of predicted patch keys with the gold patch keys.

Let:

P = unique predicted `(echo_id, lane_id)` keys

G = gold patch keys

Then:

KeyPrecision = |P ∩ G| / |P|

KeyRecall = |P ∩ G| / |G|

PatchKeyF1 = 2 × KeyPrecision × KeyRecall / (KeyPrecision + KeyRecall)

If the prediction contains no valid patch keys while gold repairs exist:

PatchKeyF1 = 0

The dataset-level value:

K

is the arithmetic mean of sample-level PatchKeyF1.

This component measures whether the model found the correct repair locations in latent lane space.

### OLD Span F1

For every gold patch key that also appears in the prediction, compare the predicted OLD span with the gold OLD span.

Tokenization uses:

- Unicode normalization.
- Whitespace splitting.
- Case-insensitive token comparison.

Tokens are compared as multisets.

For one matched patch key:

OldPrecision = matched_old_tokens / predicted_old_tokens

OldRecall = matched_old_tokens / gold_old_tokens

OLDTokenF1 is their harmonic mean.

Predicted keys that are not gold keys contribute zero.

Missing gold keys also contribute zero.

The sample-level OLD Span F1 is the arithmetic mean over the union of predicted and gold patch keys.

The dataset-level value is:

O

OLD Span F1 rewards precise localization of the visible disagreement.

### NEW Span F1

The NEW span is evaluated in the same way.

For every patch key, compare the predicted replacement text with the gold replacement text using case-insensitive token multiset F1.

The sample-level score averages over the union of predicted and gold keys.

The dataset-level value is:

N

This component rewards recovery of the intended payload, even when exact casing or one token is imperfect.

### Ledger Order LCS

The gold patch sequence is ordered by:

1. Echo ID.
2. Left-to-right repair position within that echo.

Ignore OLD and NEW text.

Convert each patch to its key:

E01/L018

E03/L041

...

Let:

S_pred = predicted patch-key sequence

S_gold = gold patch-key sequence

Compute their longest common subsequence.

Then:

LedgerLCS = LCS_length / max(1, number_of_gold_patches)

The value is clipped to 1.

The dataset-level value is:

L

This component rewards recovery of the correct sparse repair route even when some lexical details are imperfect.

### Replay Fidelity

The patch ledger is also evaluated by what happens when it is executed.

For every echo:

1. Start from the original public echo text.
2. Read predicted patches for that echo in ledger order.
3. For each patch, search for the normalized OLD span in the current echo.
4. Replace the first unmatched exact occurrence with the predicted NEW span.
5. If the OLD span cannot be found, that patch does not modify the echo.
6. Continue until all predicted patches for the echo have been attempted.

### Canonical Repaired Echoes

The canonical comparison text is not produced by a hidden model, external translator, retrieval system, or undisclosed normalization rule.

For every sample and every echo, the canonical repaired echo is defined mechanically:

1. Begin with the exact public echo string.
2. Take the gold patches for that echo from the hidden gold ledger.
3. Apply those gold patches in their published ledger order.
4. Use the resulting string as the canonical repaired echo.

An echo with no gold patch is its own canonical repaired echo.

This definition means participants can reproduce Replay Fidelity exactly on any local validation row for which they know the gold ledger.

No additional semantic annotation is consulted during scoring.

After replaying the participant prediction, compare the resulting echo with the canonical repaired echo using normalized token edit similarity.

For one echo:

ReplaySimilarity = 1 - token_edit_distance / max(1, gold_token_count, predicted_token_count)

where:

- `gold_token_count` is the number of tokens in the canonical repaired echo.
- `predicted_token_count` is the number of tokens after predicted replay.
- `token_edit_distance` is ordinary Levenshtein edit distance over whitespace-normalized tokens.

The value is clipped to:

[0, 1]

The sample-level Replay Fidelity is the arithmetic mean across E01 through E04.

The dataset-level value is:

R

This component is important because a patch can contain plausible local pieces but still fail to repair the visible echo when executed.

### Exact Ledger

Normalize ordinary whitespace in the full predicted ledger and gold ledger.

For one sample:

ExactLedger = 1

when the normalized strings are identical.

Otherwise:

ExactLedger = 0

The dataset-level value is:

X

Exact match is a completion bonus rather than the sole metric.

### Final Score

Let:

- K = Patch Key F1
- O = OLD Span F1
- N = NEW Span F1
- L = Ledger Order LCS
- R = Replay Fidelity
- X = Exact Ledger

First define the localization term:

Localization = sqrt(K × O)

Then define the repair core:

Core = 0.34 × Localization + 0.31 × N + 0.20 × R + 0.15 × L

Define the replay gate:

ReplayGate = 0.72 + 0.28 × R

Define the completion bonus:

Completion = 0.86 + 0.14 × X

The final score is:

Parallax Ledger Score = 100 × Core^1.20 × ReplayGate × Completion

The result is clipped to:

[0.01, 100]

A perfect submission has:

- K = 1
- O = 1
- N = 1
- L = 1
- R = 1
- X = 1

Therefore:

- Localization = 1
- Core = 1
- ReplayGate = 1
- Completion = 1
- Final Score = 100

The exponent of 1.20 makes the upper leaderboard region require broad competence.

The localization term requires both the correct latent repair key and the correct visible OLD span.

The NEW span term rewards actual correction content.

Replay Fidelity checks whether the ledger works when executed.

Ledger LCS rewards correct sparse ordering.

Exact Ledger adds a small premium for complete end-to-end reconstruction.

### Reproducing the Metric Locally

For every validation sample:

1. Parse the predicted patch ledger.
2. Parse the gold patch ledger.
3. Extract unique `(echo_id, lane_id)` patch keys.
4. Compute Patch Key F1.
5. Match predicted and gold records by patch key.
6. Compute OLD span token F1 over the union of keys.
7. Compute NEW span token F1 over the union of keys.
8. Convert each ledger to its ordered patch-key sequence.
9. Compute longest common subsequence divided by the number of gold patches.
10. Replay the predicted ledger against each public echo.
11. Compare each replayed echo with its canonical repaired echo using token edit similarity.
12. Average Replay Fidelity across the four echoes.
13. Compare the complete normalized ledger string for Exact Ledger.
14. Average K, O, N, L, R, and X across validation samples.
15. Compute `Localization = sqrt(K × O)`.
16. Compute `Core = 0.34 × Localization + 0.31 × N + 0.20 × R + 0.15 × L`.
17. Compute `ReplayGate = 0.72 + 0.28 × R`.
18. Compute `Completion = 0.86 + 0.14 × X`.
19. Compute `100 × Core^1.20 × ReplayGate × Completion`.
20. Clip to `[0.01, 100]`.

The provided `grader.py` implements this procedure directly.

### Intended Learned Approaches

A useful model should treat the entire sample as one multi-view source sequence.

A practical source serialization may resemble:

<ANCHOR> set an alarm for 7 tomorrow morning

<E01> kal subah 8 baje alarm laga do

<E02> kal subah 7 baje alarm laga do

<E03> parso subah 7 baje alarm laga do

<E04> kal subah 7 baje alarm set karo

The target is the patch ledger.

### Compact Encoder-Decoder Fine-Tuning

A practical baseline can fine-tune a compact pretrained seq2seq checkpoint on:

anchor + four echoes -> repair_target

Useful choices include:

- Small text-to-text transformers.
- Compact multilingual encoder-decoders.
- Byte-level models.
- Small models with added patch-grammar tokens.

### Structural Tokens

Participants may add target grammar symbols as tokenizer tokens:

- `<PATCH>`
- `</PATCH>`
- `<OLD>`
- `</OLD>`
- `<NEW>`
- `</NEW>`
- Echo IDs.
- Lane IDs.

This can reduce unnecessary fragmentation of the output language.

### Cross-View Attention

A strong model should compare all views rather than encode each echo independently without interaction.

Useful learned designs include:

- One concatenated encoder source.
- Segment embeddings for anchor and echo identity.
- Per-view encoding followed by learned pooling.
- Pairwise anchor-echo interaction layers.
- Echo-to-echo attention.
- Compact cross-view reranking heads.

### Copy-Aware Decoding

OLD spans are visible in public echoes.

NEW spans are often supported by another view.

Useful mechanisms include:

- Ordinary cross-attention copying.
- Pointer-style heads.
- Copy gates.
- Source alignment supervision derived from training targets.
- Span-copy auxiliary losses.

### Lane Auxiliary Tasks

Participants may derive extra supervision from `repair_target`.

Examples include:

- Lane presence.
- Echo repair count.
- Per-echo lane set.
- Total patch count.
- OLD span alignment.
- NEW span source-view alignment.
- Patch-key sequence.
- Whether a lane is repaired in more than one echo.
- Pairwise lane co-repair prediction.

These labels are allowed because they are derived entirely from released challenge training targets.

### Constrained Decoding

The patch grammar is regular enough to constrain.

Useful constraints include:

- `<OLD>` cannot appear before a legal echo and lane.
- Every `<PATCH>` must close.
- Echo IDs must be E01 through E04.
- Lane IDs must come from the training inventory.
- `<NEW>` must follow `</OLD>`.
- End-of-sequence can be blocked while a patch is open.

These restrictions improve serialization without hand-coding the semantic answer.

### Hard-Negative Learning

The public echoes contain plausible but conflicting payloads.

This creates natural hard negatives.

A participant may derive training pairs such as:

- Correct lane versus incorrect lane.
- Correct echo versus unmodified echo.
- Gold replacement versus rival payload from another view.
- Required patch versus unnecessary patch.

A compact model may benefit from an auxiliary ranking objective over these alternatives.

### Local Validation

A random validation split can be misleading.

A stronger local estimate groups by:

- Exact repair constellation.
- Lane co-repair pairs.
- Patch count.
- Echo repair pattern.
- Rare lane combinations.

Keep all derived versions of one sample together.

Do not split augmented copies, cached features, or auxiliary labels from one sample across local train and validation.

### What Makes the Challenge Hard

Several sources of uncertainty interact.

### No Globally Trusted Echo

Different views can be wrong in different places.

A model cannot simply select E01 or E04 as the canonical sentence.

### Cross-Language Evidence

The English anchor and mixed-language echoes can express the same payload with different surface forms.

### Plausible Drift

Incorrect values are chosen to remain locally believable.

Surface fluency is therefore a weak signal.

### Sparse Output

Most source text is already correct.

The model must learn when **not** to emit a repair.

### Opaque Lane Semantics

The correct repair lane must be inferred from training examples.

### Multi-Patch Coupling

Several disagreements can occur in one sample.

A wrong global interpretation can cause multiple patch errors.

### Replay Requirement

Plausible-looking patches still need to execute successfully against the public text.

### Constellation Holdout

The evaluation split recombines familiar repair behaviors in disagreement patterns that are not repeated as full training targets.

### CPU Constraint

Very large sequence models are impractical.

Strong results require efficient use of compact pretrained models.

### Practical CPU Baseline

A practical eligible baseline may:

- Read `train.jsonl`.
- Serialize anchor and echoes with explicit segment tokens.
- Discover lane IDs from training targets.
- Extend a compact pretrained tokenizer with patch grammar tokens.
- Fine-tune a small encoder-decoder on complete repair ledgers.
- Use a local constellation-aware validation split.
- Decode with a small beam.
- Apply grammar constraints.
- Evaluate with the official metric.
- Generate one repair ledger for every test sample.

A stronger system may add:

- Copy-aware decoding.
- Patch-count auxiliary prediction.
- Per-echo repair detection.
- Lane-presence auxiliaries.
- Cross-view contrastive objectives.
- Hard-negative ranking among rival payloads.
- Two compact models with different tokenization schemes.
- Model-based reranking using replay validity.

### Allowed Resources

Participants may use:

- Released challenge files.
- Generic pretrained sequence-to-sequence checkpoints.
- Generic pretrained tokenizers distributed with eligible checkpoints.
- Standard machine-learning libraries.
- Standard deep-learning frameworks.
- Public architecture implementations.
- Beam search.
- Grammar-constrained decoding.
- Tokenizer extension with released structural tokens.
- Auxiliary labels derived from released training targets.
- Participant-created local validation splits.
- CPU-friendly ensembles of eligible learned models.

### Disallowed Resources

Participants may not use:

- A checkpoint trained specifically for this challenge.
- External examples containing challenge repair ledgers.
- External mappings from lane IDs to descriptive meanings.
- Search engines at inference time.
- External retrieval at inference time.
- Hand-written keyword-to-lane rules.
- Manual lane dictionaries.
- Rule-only repair systems.
- TF-IDF retrieval as the primary predictor.
- Nearest-neighbor ledger replay as the primary predictor.
- Manual annotation of test samples.
- Hidden evaluator files.
- Hard-coded test repairs.
- Submission-feedback reconstruction of hidden answers.
- `sample_id` as a predictive feature.
- Row order as a predictive feature.
- Filename order as a predictive feature.

### Leakage Rules

The atomic modeling unit is one complete parallax sample:

- One anchor.
- Four echoes.
- One repair ledger.

All derivatives of one sample should remain together in local validation.

This includes:

- Tokenized copies.
- Span alignments.
- Copy labels.
- Lane-presence labels.
- Patch-count labels.
- Hard-negative variants.
- Cached embeddings.
- Augmented source serializations.

Do not use packaging artifacts as features.

### Limitations

The benchmark focuses on short assistant-style requests.

It does not measure:

- Open-domain conversation.
- Long-form translation.
- Document revision.
- Factual web retrieval.
- General spelling correction.
- Speech recognition.
- Acoustic code-switching.
- Long-context memory.
- Arbitrary software patching.

Mixed-language Romanization is variable.

Some payloads are easier to align than others.

Some replacement strings may have several plausible surface realizations, but each released training row uses one deterministic target repair.

The metric provides partial lexical credit, but it does not automatically treat all paraphrases as equivalent.

### Expected Outcome

A successful system should:

- Fuse evidence from one anchor and four partially trustworthy echoes.
- Detect semantically inconsistent payloads.
- Avoid unnecessary edits.
- Recover correct replacement text.
- Induce the hidden repair-lane vocabulary.
- Serialize a minimal patch program.
- Generalize to unfamiliar multi-repair constellations.
- Produce patches that successfully replay against the public echoes.
- Train and infer efficiently on 10 CPU cores and 62.5 GiB RAM.

The prediction objective is:

**reconstruct the minimal parallax patch ledger that reconciles every test sample.**
