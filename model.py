from collections import Counter


def make_model_state():
    """Create and return a fresh model state dictionary.

    The state contains simple containers for unigram and bigram counts and
    a `trained` flag. Passing this state explicitly avoids module-level
    globals and keeps the API functional-style.
    """
    return {
        'unigrams': Counter(),
        'bigrams': Counter(),
        'trained': False
    }


def _tokenize(text):
    """Very small tokenizer: split lines, whitespace-tokenize, strip basic punctuation.

    Returns a list of token lists (sentences), each wrapped with <s> and </s>.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    sents = []
    for s in lines:
        toks = [t.lower().strip('.,!?;:\"()') for t in s.split() if t]
        toks = ['<s>'] + toks + ['</s>']
        sents.append(toks)
    return sents


def train_from_text(state, text):
    """Update `state` in-place by counting unigrams and bigrams from `text`.

    Args:
        state: dict created by `make_model_state()`.
        text: full text (str) to tokenize and count.

    The function mutates `state` and sets the `trained` flag to True.
    """
    sents = _tokenize(text)
    unigrams = state['unigrams']
    bigrams = state['bigrams']

    for toks in sents:
        for i in range(len(toks)):
            unigrams[toks[i]] += 1
        for i in range(len(toks) - 1):
            bigrams[(toks[i], toks[i+1])] += 1

    state['trained'] = True


def is_trained(state):
    """Return True if the model `state` has been trained."""
    return bool(state.get('trained'))


def vocab_size(state):
    """Return vocabulary size (number of distinct unigrams) in `state`."""
    return len(state['unigrams'])


def p_bigram_laplace(state, word, prec, vocab_sz=None):
    """Compute P(word | prec) with Laplace smoothing using counts in `state`.

    If `vocab_sz` is not provided, it is inferred from the state's unigrams.
    """
    if vocab_sz is None:
        vocab_sz = vocab_size(state)
    c_bigram = state['bigrams'].get((prec, word), 0)
    c_prec = state['unigrams'].get(prec, 0)
    return (c_bigram + 1) / (c_prec + vocab_sz)


def generate_greedy(state, start_token, max_len=20):
    """Greedy decoding using bigram probabilities from `state`.

    Args:
        state: model state dict
        start_token: optional prompt string; the last token is used as context
        max_len: maximum number of tokens to generate

    Returns a single string with generated tokens (space-separated).
    """
    if not start_token:
        prev = '<s>'
        generated = []
    else:
        toks = [t.lower().strip('.,!?;:\"()') for t in start_token.split() if t]
        prev = toks[-1] if toks else '<s>'
        generated = toks.copy()

    # Build candidate probabilities for the given prev token and pick the max
    for _ in range(max_len):
        candidates = {}
        for (w1, w2), _count in state['bigrams'].items():
            if w1 == prev:
                candidates[w2] = p_bigram_laplace(state, w2, prev)
        if not candidates:
            # no observed continuation from this token
            break
        # greedy selection: highest probability next token
        next_word = max(candidates.items(), key=lambda x: x[1])[0]
        if next_word == '</s>':
            break
        generated.append(next_word)
        prev = next_word

    return ' '.join(generated)

