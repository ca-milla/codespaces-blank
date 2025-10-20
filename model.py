import random
import math


def make_model_state():
    """Create and return a fresh model state dictionary.

    The model state is a plain Python dict with the following keys:
      - 'unigrams': mapping token -> count
      - 'bigrams': mapping (prev, next) -> count
      - 'trigrams': mapping (prev2, prev1, next) -> count
      - 'trained': boolean flag set to True after training

    Using simple dicts keeps the implementation tiny and dependency-free.
    Callers are expected to pass this state explicitly to the training and
    generation functions (no module-level globals).
    """
    # Use plain dicts for portability (no Counter required).
    # Values are integer counts; callers should treat them as mappings.
    return {
        'unigrams': {},
        'bigrams': {},
        'trigrams': {},
        'trained': False
    }


def tokenize(text):
    """Very small tokenizer.

    Behavior:
      - splits the input text into non-empty lines
      - each line is split on whitespace into tokens
      - basic punctuation is stripped from token edges
      - tokens are lower-cased
      - each sentence (line) is wrapped with special tokens '<s>' and '</s>' to
        indicate sentence start and end for language modeling.

    Returns:
      list of token lists (one list per sentence)
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    sents = []
    for s in lines:
        toks = [t.lower().strip('.,!?;:"()') for t in s.split() if t]
        toks = ['<s>'] + toks + ['</s>']
        sents.append(toks)
    return sents


def train_from_text(state, text):
    """Train/update `state` in-place from raw text.

    This function tokenizes the input into sentences and updates three count
    tables (unigrams, bigrams, trigrams). Counts are incremented; the function
    mutates the provided `state` and sets `state['trained'] = True` when done.

    Notes:
      - The tokenizer wraps sentences with '<s>' and '</s>' so those tokens
        are also counted in unigrams/bigrams/trigrams.
      - Using counts allows simple Laplace-smoothed probability estimates later.

    Args:
      state: dict returned by `make_model_state()`
      text: string with the training text (can contain multiple lines)
    """
    sents = tokenize(text)
    unigrams = state['unigrams']
    bigrams = state['bigrams']
    trigrams = state['trigrams']

    for toks in sents:
        for i in range(len(toks)):
            tok = toks[i]
            unigrams[tok] = unigrams.get(tok, 0) + 1
        for i in range(len(toks) - 1):
            pair = (toks[i], toks[i+1])
            bigrams[pair] = bigrams.get(pair, 0) + 1
        for i in range(len(toks) - 2):
            trip = (toks[i], toks[i+1], toks[i+2])
            trigrams[trip] = trigrams.get(trip, 0) + 1

    state['trained'] = True


def is_trained(state):
    """Return True if the model `state` has been trained."""
    return bool(state.get('trained'))


def vocab_size(state):
    """Return vocabulary size (number of distinct unigram types).

    This value is used as V in Laplace smoothing: P(w|context) = (count + 1) / (context_count + V)
    """
    return len(state['unigrams'])


def p_bigram_laplace(state, word, prec, vocab_sz=None):
    """Compute smoothed bigram probability P(word | prec).

    Uses add-one (Laplace) smoothing. If `vocab_sz` is omitted it is inferred
    from the number of distinct unigrams in the state.

    Args:
      state: model state dict
      word: candidate next token
      prec: previous token (context)
      vocab_sz: optional vocabulary size (int)

    Returns:
      float probability in (0,1]
    """
    if vocab_sz is None:
        vocab_sz = vocab_size(state)
    c_bigram = state['bigrams'].get((prec, word), 0)
    c_prec = state['unigrams'].get(prec, 0)
    return (c_bigram + 1) / (c_prec + vocab_sz)


def p_trigram_laplace(state, word, prec2, prec1, vocab_sz=None):
    """Compute smoothed trigram probability P(word | prec2, prec1).

    The denominator uses the bigram count for (prec2, prec1) so the estimate
    is P(w | prev2, prev1) = (C(prev2,prev1,w) + 1) / (C(prev2,prev1) + V)

    Args:
      state: model state dict
      word: candidate next token
      prec2: token at position t-2
      prec1: token at position t-1
      vocab_sz: optional vocabulary size (int)

    Returns:
      float probability in (0,1]
    """
    if vocab_sz is None:
        vocab_sz = vocab_size(state)
    c_trigram = state['trigrams'].get((prec2, prec1, word), 0)
    c_bigram = state['bigrams'].get((prec2, prec1), 0)
    return (c_trigram + 1) / (c_bigram + vocab_sz)


def generate_greedy(state, start_token, max_len=20, prefer_trigram=True):
    """Greedy decoding: pick the most likely next token at each step.

    The generator builds candidates using trigram context when available and
    preferred; otherwise it uses bigram context. It always performs a greedy
    (argmax) selection of the next token's probability.

    Args:
      state: model state dict
      start_token: optional prompt string; tokenization is simple whitespace
      max_len: maximum number of tokens to append (stops early on '</s>' or no candidates)
      prefer_trigram: if True, attempt trigram expansions when two-token context exists

    Returns:
      a single string with generated tokens (space-separated)
    """
    if not start_token:
        prev = '<s>'
        prev2 = None
        generated = []
    else:
        toks = [t.lower().strip('.,!?;:"()') for t in start_token.split() if t]
        if toks:
            generated = toks.copy()
            if len(toks) >= 2:
                prev2 = toks[-2]
                prev = toks[-1]
            else:
                prev2 = None
                prev = toks[-1]
        else:
            prev = '<s>'
            prev2 = None
            generated = []

    # Build candidate probabilities for the given prev token and pick the max
    for _ in range(max_len):
        candidates = {}
        # Prefer trigram context when requested and available (prev2, prev -> next)
        if prefer_trigram and prev2 is not None:
            for (a, b, c), _count in state['trigrams'].items():
                if a == prev2 and b == prev:
                    candidates[c] = p_trigram_laplace(state, c, prev2, prev)
        # If no trigram candidates (or trigram not preferred), fall back to bigrams (prev -> next)
        if not candidates:
            for (w1, w2), _count in state['bigrams'].items():
                if w1 == prev:
                    candidates[w2] = p_bigram_laplace(state, w2, prev)
        if not candidates:
            # no observed continuation from this token/context
            break
        # greedy selection: highest probability next token
        next_word = max(candidates.items(), key=lambda x: x[1])[0]
        if next_word == '</s>':
            break
        generated.append(next_word)
        # shift context
        prev2 = prev
        prev = next_word

    return ' '.join(generated)


def generate_sample(state, start_token, max_len=20, prefer_trigram=True, top_k=0, top_p=0.0, temperature=1.0):
    """Stochastic decoding with optional top-k / top-p (nucleus) and temperature.

    Behavior:
      - builds candidate distribution from trigram or bigram context
      - optionally filters candidates with top-k and/or top-p
      - applies temperature to control randomness
      - samples the next token according to the resulting weights

    Args:
      state: model state dict
      start_token: optional prompt string
      max_len: maximum tokens to generate
      prefer_trigram: prefer trigram context when available
      top_k: if >0, keep only the top_k highest-probability candidates
      top_p: if in (0,1), keep the smallest set of top tokens with cumulative prob >= top_p
      temperature: >0 float, 1.0 means no scaling; lower values make distribution sharper

    Returns:
      generated string (space-separated tokens)
    """
    if not start_token:
        prev = '<s>'
        prev2 = None
        generated = []
    else:
        toks = [t.lower().strip('.,!?;:"()') for t in start_token.split() if t]
        if toks:
            generated = toks.copy()
            if len(toks) >= 2:
                prev2 = toks[-2]
                prev = toks[-1]
            else:
                prev2 = None
                prev = toks[-1]
        else:
            prev = '<s>'
            prev2 = None
            generated = []

    for _ in range(max_len):
        candidates = {}
        if prefer_trigram and prev2 is not None:
            for (a, b, c), _count in state['trigrams'].items():
                if a == prev2 and b == prev:
                    candidates[c] = p_trigram_laplace(state, c, prev2, prev)
        if not candidates:
            for (w1, w2), _count in state['bigrams'].items():
                if w1 == prev:
                    candidates[w2] = p_bigram_laplace(state, w2, prev)
        if not candidates:
            break
        # apply top-k / top-p filtering if requested
        items = list(candidates.items())
        # items: list of (word, prob)
        # sort descending by prob
        items.sort(key=lambda x: x[1], reverse=True)

        # top-k
        if top_k and top_k > 0 and top_k < len(items):
            items = items[:top_k]

        # top-p (nucleus)
        if top_p and 0.0 < top_p < 1.0:
            cum = 0.0
            kept = []
            for w, p in items:
                cum += p
                kept.append((w, p))
                if cum >= top_p:
                    break
            if kept:
                items = kept

        words = [w for w, p in items]
        probs = [p for w, p in items]

        # apply temperature: convert probs -> logits, scale, back to weights
        if temperature is None or temperature <= 0:
            temperature = 1.0
        # use logs to avoid extreme numbers
        logits = [math.log(p) for p in probs]
        scaled = [math.exp(l / temperature) for l in logits]
        # normalize
        total = sum(scaled)
        if total <= 0:
            weights = [1.0 / len(scaled)] * len(scaled)
        else:
            weights = [s / total for s in scaled]

        next_word = random.choices(words, weights=weights, k=1)[0]
        if next_word == '</s>':
            break
        generated.append(next_word)
        prev2 = prev
        prev = next_word

    return ' '.join(generated)


def beam_search(state, start_token, beam_size=3, steps=20, prefer_trigram=True):
    """Beam search over sequences.

    Implementation notes:
      - beams are tuples (log_score, sequence)
      - for numerical stability we accumulate log-probabilities (sum of log p)
      - in each step we expand each beam with all valid next tokens using
        trigram expansions first (if preferred) then bigram expansions
      - after generating candidates we keep only the top `beam_size` beams

    Returns:
      a list of (log_score, sequence) sorted by log_score descending
    """
    # initialize
    if not start_token:
        start_tok = '<s>'
    else:
        toks = [t.lower().strip('.,!?;:"()') for t in start_token.split() if t]
        start_tok = toks[-1] if toks else '<s>'

    beams = [(0.0, [start_tok])]
    for _ in range(steps):
        candidates = []
        for score, seq in beams:
            prev = seq[-1]
            prev2 = seq[-2] if len(seq) >= 2 else None
            # build next candidates
            if prefer_trigram and prev2 is not None:
                for (a, b, c), _count in state['trigrams'].items():
                    if a == prev2 and b == prev:
                        p = p_trigram_laplace(state, c, a, b)
                        if p > 0:
                            candidates.append((score + math.log(p), seq + [c]))
            # bigram expansions
            for (w1, w2), _count in state['bigrams'].items():
                if w1 == prev:
                    p = p_bigram_laplace(state, w2, w1)
                    if p > 0:
                        candidates.append((score + math.log(p), seq + [w2]))
        if not candidates:
            break
        # keep top beam_size by score (log-probability)
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_size]
    return beams

