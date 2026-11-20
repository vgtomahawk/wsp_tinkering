I edited the attention output projection (o_proj) and MLP output projection (down_proj) in layers 8–25, 36 matrices in total. The edit removes about 87% of what these blocks write along a 3-dimensional "apologise / inability / decline" direction. I could not run the model at all (0 forward passes), so neither the drop in refusals nor the lack of damage has been tested. This is a reasoned bet from the weights alone. Its weak point is the size of the weight changes, larger than I wanted (mean 11%, up to 21% per matrix).

**What the weights showed**
- **No refusal trigger could be found.** I looked for it in several ways:
  - Neurons that write refusal tokens do not share an input direction.
  - Neurons that read harmful-word embeddings showed only weak selectivity.
  - The most promising attention head, layer 22 head 3, turned out to copy "unable / refuse / lack" ideas already in the text forward, rather than detect harm.
  - The strongest neurons for writing the first word "I" (layer 22 neuron 2774, z = 20) serve first-person speech in general, in many languages, and do not respond to the chat-template tokens.
- **The output side was clean.** I defined three directions in token space, built from these word groups:
  - apology: sorry, apologize, apologies;
  - inability: cannot, unable, inability, incapable;
  - refusal: refuse, decline, refusal, unwilling, deny.

  I removed from these directions everything shared with about 40 common words: answer openers like "Okay" and "Here", pronouns, and function words like " not", " can", " will", " the".

  The result is specific to refusal language:
  - The 40 tokens most affected are all refusal, apology or denial words.
  - Every one of the 40 common words has exactly zero overlap with it, and the typical token overlaps by 0.6%.
  - Related ordinary words are barely affected: " failed" 9%, " error" 4%.
- **It is written by a handful of neurons.** In layers 13–22 these include layer 17 neuron 6841 and layer 15 neuron 3461 ("cannot / unable"), and layer 22 neuron 3828 ("refusal / deny").

**The edit**
- Each block's output passes through a normalisation that scales dimensions very unevenly; some scales are 0 and some reach about 260.
  - A plain projection would have spread 75% of the change outside the refusal directions.
  - An exact correction would have needed weight changes of up to 34%.
- I used a regularised version (λ = 3) that keeps the change aimed at the refusal directions. It removes 87% of the write on average (at least 54% in every matrix).
- Layers 0–7 are untouched, because there these directions mostly carry word identity, which the model needs to read such words in a prompt.

**Why it should help, and the risks**
- **Expected benefit:** the model can no longer push its output toward "sorry", "cannot", "unable" or "refuse". Ordinary answer openers are mathematically unaffected, so the next-token change on ordinary requests should be small.
- **Refusals may only partly disappear.**
  - The decision to refuse was not located. The model can still start its reply with "I", and can still refuse with ordinary words ("I can't", "I will not"), which this edit does not block.
  - Only part of each refusal word's signal lies in these directions (for example 23% for " cannot", 46% for " unable").
- **Ordinary text may suffer somewhat.**
  - Answers that genuinely use words like "cannot" or "sorry" become a little less likely.
  - The weight changes may slightly alter the size of each block's output after normalisation.

FINAL