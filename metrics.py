import evaluate 
from nltk.tokenize import word_tokenize
from collections import Counter 
import re


try:
    bleu_metric = evaluate.load("bleu") # Thay đổi từ datasets.load_metric sang evaluate.load
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")
    cider_metric = evaluate.load("cider") # Đối với CIDEr, bạn có thể cần phiên bản tùy chỉnh hoặc pycocoevalcap
    # F1/Recall Token-based sẽ được tính toán thủ công
except ImportError:
    print("Warning: Some evaluation metrics might not be installed. Please run 'pip install evaluate datasets sacrebleu rouge_score meteor pycocoevalcap' and ensure nltk is installed.")
    bleu_metric, rouge_metric, meteor_metric, cider_metric = None, None, None, None


def normalize_text(text):
    """
    Standardize text for tokenization and comparison.
    - Convert to lowercase
    - Remove punctuation
    - Normalize Vietnamese characters (optional, but good for consistency)
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip() # Replace multiple spaces with single space
    # You might want more sophisticated Vietnamese normalization if needed
    return text

def calculate_f1_recall_token_level(predictions, references, vocab):
    """
    Calculates token-level F1 and Recall.
    predictions: dict {image_id: predicted_caption_text}
    references: dict {image_id: [list of ground_truth_caption_texts]}
    vocab: The Vocabulary object with its tokenizer
    """
    total_f1 = 0.0
    total_recall = 0.0
    num_samples = 0

    for image_id, predicted_caption in predictions.items():
        if image_id not in references:
            continue # Skip if no references available for this image_id

        num_samples += 1
        
        # Normalize and tokenize predicted caption using your model's tokenizer
        pred_tokens = vocab.tokenizer.tokenize(normalize_text(predicted_caption))
        
        # Get all reference captions for this image_id
        ref_captions = references[image_id]
        
        best_recall_for_sample = 0.0

        for ref_caption_text in ref_captions:
            ref_tokens = vocab.tokenizer.tokenize(normalize_text(ref_caption_text))

            if not pred_tokens or not ref_tokens:
                continue # Skip if either token list is empty

            # Calculate token counts
            pred_counts = Counter(pred_tokens)
            ref_counts = Counter(ref_tokens)

            # True Positives: common tokens
            common_tokens = pred_counts & ref_counts
            tp = sum(common_tokens.values())

            # Precision: (True Positives) / (Predicted Tokens)
            precision = tp / sum(pred_counts.values()) if sum(pred_counts.values()) > 0 else 0.0

            # Recall: (True Positives) / (Reference Tokens)
            recall = tp / sum(ref_counts.values()) if sum(ref_counts.values()) > 0 else 0.0
            
            # F1 Score
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # For recall, we want the max recall across all ground truths for a single prediction
            best_recall_for_sample = max(best_recall_for_sample, recall)
            total_f1 += f1 # You might want to average F1 across references or take the best F1 too

        total_recall += best_recall_for_sample # Accumulate the best recall for this sample
        # For F1, a common approach is to compute one F1 per (pred, ref) pair, then average.
        # Here, it's summed for simplicity, consider if you need a different averaging strategy.

    avg_f1 = total_f1 / num_samples if num_samples > 0 else 0.0
    avg_recall = total_recall / num_samples if num_samples > 0 else 0.0
    
    return avg_f1, avg_recall

def evaluate_metrics(predictions, references, vocab):
    """
    Calculates various captioning metrics.
    predictions: dict {image_id: predicted_caption_text}
    references: dict {image_id: [list of ground_truth_caption_texts]}
    vocab: The Vocabulary object with its tokenizer
    """
    results = {}

    # Prepare data for huggingface evaluate metrics
    # The 'evaluate' library expects lists of strings or lists of lists of strings
    # For BLEU, ROUGE, METEOR:
    # predictions_list = [pred_text for img_id, pred_text in predictions.items()]
    # references_list_of_lists = [[gt_text for gt_text in refs] for img_id, refs in references.items() if img_id in predictions]
    
    # Ensure consistent order of image_ids for pairing
    common_image_ids = sorted(list(set(predictions.keys()) & set(references.keys())))

    # Filter out empty entries to prevent errors in metric calculation
    # Ensure that references_for_metric are lists of lists, even for a single reference
    predictions_for_metric = [predictions[img_id] for img_id in common_image_ids]
    references_for_metric = [[ref for ref in references[img_id]] for img_id in common_image_ids]
    
    # Filter out any entries where prediction or all references are empty after tokenization
    # This can happen if normalization results in empty strings.
    filtered_predictions = []
    filtered_references = []
    for pred, refs_list in zip(predictions_for_metric, references_for_metric):
        # We need at least one non-empty reference
        valid_refs = [r for r in refs_list if normalize_text(r)]
        if normalize_text(pred) and valid_refs:
            filtered_predictions.append(pred)
            filtered_references.append(valid_refs)

    if not filtered_predictions or not filtered_references:
        print("Warning: No valid predictions or references found for metric calculation after filtering empty strings.")
        return {
            'BLEU': 0.0, 'ROUGE': 0.0, 'METEOR': 0.0, 'CIDEr': 0.0,
            'F1_token_avg': 0.0, 'Recall_token_avg': 0.0
        }

 
    # Bleu expects references as a list of lists: [[ref1, ref2], [ref3]]
    # And predictions as a list of strings: ["pred1", "pred2"]
    # Ensure that each reference in filtered_references is itself a list of strings.
    # The way filtered_references is constructed above already makes it a list of lists of strings.
    bleu_score = bleu_metric.compute(predictions=filtered_predictions, references=filtered_references)
    results['BLEU'] = bleu_score['bleu'] * 100 if 'bleu' in bleu_score else 0.0 # Convert to percentage


    rouge_score = rouge_metric.compute(predictions=filtered_predictions, references=filtered_references)
    results['ROUGE'] = rouge_score['rougeL'] * 100 if 'rougeL' in rouge_score else 0.0 # Using rougeL for simplicity


    # Meteor expects predictions as list of strings and references as list of lists of strings
    meteor_score = meteor_metric.compute(predictions=filtered_predictions, references=filtered_references)
    results['METEOR'] = meteor_score['meteor'] * 100 if 'meteor' in meteor_score else 0.0


    # CIDEr also expects predictions as list of strings and references as list of lists of strings
    cider_score = cider_metric.compute(predictions=filtered_predictions, references=filtered_references)
    results['CIDEr'] = cider_score['cider'] * 100 if 'cider' in cider_score else 0.0
    
    # 5. Token-level F1 and Recall (custom calculation)
    # The `calculate_f1_recall_token_level` function already expects dictionaries
    avg_f1, avg_recall = calculate_f1_recall_token_level(predictions, references, vocab) 
    results['F1_token_avg'] = avg_f1 * 100
    results['Recall_token_avg'] = avg_recall * 100

    return results