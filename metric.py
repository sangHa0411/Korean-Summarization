import re
import six
import numpy as np
from tqdm import tqdm
from konlpy.tag import Mecab
from rouge_score import scoring
from rouge_utils import (
    create_ngrams, 
    score_lcs, 
    score_ngrams, 
)

class KoreanRougeScorer(scoring.BaseScorer):
    def __init__(self, rouge_types):
        self.rouge_types = rouge_types
        self.tokenizer = Mecab()

    def score(self, target, prediction):
        if len(self.rouge_types) == 1 and self.rouge_types[0] == "rougeLsum":
            target_tokens = None
            prediction_tokens = None
        else:
            target_tokens = self.tokenizer.morphs(target)
            prediction_tokens = self.tokenizer.morphs(prediction) 
        result = {}

        for rouge_type in self.rouge_types:
            if rouge_type == "rougeL":
                scores = score_lcs(target_tokens, prediction_tokens)
            elif re.match(r"rouge[0-9]$", six.ensure_str(rouge_type)):
                # Rouge from n-grams.
                n = int(rouge_type[5:])
                if n <= 0:
                    raise ValueError("rougen requires positive n: %s" % rouge_type)
            
                target_ngrams = create_ngrams(target_tokens, n)
                prediction_ngrams = create_ngrams(prediction_tokens, n)
                scores = score_ngrams(target_ngrams, prediction_ngrams)
            else:
                raise ValueError("Invalid rouge type: %s" % rouge_type)
            result[rouge_type] = scores
        return result

def compute(predictions, references, use_agregator=True):
	rouge_types = ["rouge1", "rouge2", "rougeL"]
	scorer = KoreanRougeScorer(rouge_types=rouge_types)

	if use_agregator:
		aggregator = scoring.BootstrapAggregator()
	else:
		scores = []

	for ref, pred in zip(references, predictions):
		score = scorer.score(ref, pred)
		if use_agregator:
			aggregator.add_scores(score)
		else:
			scores.append(score)

	if use_agregator:
		result = aggregator.aggregate()
	else:
		result = {}
		for key in scores[0]:
			result[key] = list(score[key] for score in scores)
	return result


def compute_metrics(eval_pred, tokenizer):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]
    result = compute(predictions=decoded_preds, references=decoded_labels)
    
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    return {k: round(v, 4) for k, v in result.items()}

