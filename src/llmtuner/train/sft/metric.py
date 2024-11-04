from dataclasses import dataclass
from typing import Sequence, Tuple, Union, Dict
import numpy as np
from transformers import PreTrainedTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.gleu_score import sentence_gleu
from rouge import Rouge
import jieba

@dataclass
class ComputeMetrics:
    tokenizer: PreTrainedTokenizer

    def __call__(self, eval_preds: Sequence[Union[np.ndarray, Tuple[np.ndarray]]]) -> Dict[str, float]:

        preds, labels = eval_preds
        score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-1": [], "bleu-2": [], "bleu-3": [], "bleu-4": [], "gleu": [], "distinct-1": [], "distinct-2": []}

        # preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        # labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))

            # BLEU scores
            weights_for_bleu = {
                'bleu-1': (1.0, 0, 0, 0),
                'bleu-2': (0.5, 0.5, 0, 0),
                'bleu-3': (0.33, 0.33, 0.33, 0),
                'bleu-4': (0.25, 0.25, 0.25, 0.25)
            }
            smoothie = SmoothingFunction().method3
            for bleu_key, weights in weights_for_bleu.items():
                score_dict[bleu_key].append(round(sentence_bleu([reference], hypothesis, weights=weights, smoothing_function=smoothie) * 100, 4))

            # GLEU score
            gleu_score = sentence_gleu([reference], hypothesis)
            score_dict['gleu'].append(round(gleu_score * 100, 4))

            # ROUGE scores
            if len(hypothesis) == 0 or len(reference) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]
                for k, v in result.items():
                    score_dict[k].append(round(v["f"] * 100, 4))

            # DISTINCT scores
            distinct_1 = len(set(hypothesis)) / len(hypothesis) if hypothesis else 0
            distinct_2 = len(set(zip(hypothesis, hypothesis[1:]))) / (len(hypothesis) - 1) if len(hypothesis) > 1 else 0
            score_dict['distinct-1'].append(round(distinct_1 * 100, 4))
            score_dict['distinct-2'].append(round(distinct_2 * 100, 4))

        return {k: float(np.mean(v)) for k, v in score_dict.items() if v}
