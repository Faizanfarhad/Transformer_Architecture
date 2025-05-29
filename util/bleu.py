import os 
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
import math
from collections import Counter
import numpy as np

def bleu_status(hypothesis, reference):
    """Compute statistics for BLEU."""
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    for n in range(1,5):
        s_ngram = Counter(
            [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
        )
        r_ngram = Counter(
            [tuple(reference[i:i+n]) for i in range(len(reference) + 1 -n)]
        )
        stats.append(max([sum((s_ngram & r_ngram).values()), 0]))
        stats.append(max([len(hypothesis) + 1 -n , 0]))
    return stats

def bleu(stats):
    """Compute BLEU given n-gram statistics."""
    if len(list(filter(lambda x: x==0, stats))) > 0:
        return 0
    (c,r) = stats[:2]
    log_bleu_prec = sum(
        [math.log(float(x) / y) for x,y in zip(stats[2::2], stats[3::2])]
    )

    return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)

def get_blue(hypothesis, refrence):
    """Get validation BLEU score for dev set."""
    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    for hyp, ref in zip(hypothesis, refrence):
        stats += np.array(bleu_status(hyp,ref))
        return 100 * bleu(stats)

def idx_to_word(x,vocab):
    words = []
    for i in x:
        word = vocab.itos[i]
        if '<' not in word:
            words.append(word)
    words = " ".join(words)
    return words