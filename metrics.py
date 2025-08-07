import evaluate 
from nltk.tokenize import word_tokenize
from collections import Counter 
import re

# Tải các metrics riêng lẻ để xử lý lỗi tốt hơn
bleu_metric = None
rouge_metric = None
meteor_metric = None
cider_metric = None

try:
    bleu_metric = evaluate.load("bleu")
except Exception as e:
    print(f"Warning: Could not load BLEU metric. Error: {e}")

try:
    rouge_metric = evaluate.load("rouge")
except Exception as e:
    print(f"Warning: Could not load ROUGE metric. Error: {e}")

try:
    meteor_metric = evaluate.load("meteor")
except Exception as e:
    print(f"Warning: Could not load METEOR metric. Error: {e}")

try:
    cider_metric = evaluate.load("cider")
except Exception as e:
    print(f"Warning: Could not load CIDEr metric. Error: {e}")


def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def calculate_f1_recall_token_level(predictions, references, vocab):
    total_f1 = 0.0
    total_recall = 0.0
    num_samples = 0

    for image_id, predicted_caption in predictions.items():
        if image_id not in references:
            continue

        num_samples += 1

        pred_tokens = vocab.tokenizer.tokenize(normalize_text(predicted_caption))
        ref_captions = references[image_id]

        best_recall_for_sample = 0.0
        best_f1_for_sample = 0.0 # Thêm biến này

        for ref_caption_text in ref_captions:
            ref_tokens = vocab.tokenizer.tokenize(normalize_text(ref_caption_text))

            if not pred_tokens or not ref_tokens:
                continue

            pred_counts = Counter(pred_tokens)
            ref_counts = Counter(ref_tokens)

            common_tokens = pred_counts & ref_counts
            tp = sum(common_tokens.values())

            precision = tp / sum(pred_counts.values()) if sum(pred_counts.values()) > 0 else 0.0
            recall = tp / sum(ref_counts.values()) if sum(ref_counts.values()) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            best_recall_for_sample = max(best_recall_for_sample, recall)
            best_f1_for_sample = max(best_f1_for_sample, f1) # Cập nhật F1 tốt nhất

        total_recall += best_recall_for_sample
        total_f1 += best_f1_for_sample # Chỉ cộng F1 tốt nhất cho ảnh này

    avg_f1 = total_f1 / num_samples if num_samples > 0 else 0.0
    avg_recall = total_recall / num_samples if num_samples > 0 else 0.0

    return avg_f1, avg_recall

def evaluate_metrics(predictions, references, vocab):
    results = {}

    common_image_ids = sorted(list(set(predictions.keys()) & set(references.keys())))

    predictions_for_metric = [predictions[img_id] for img_id in common_image_ids]
    references_for_metric = [[ref for ref in references[img_id]] for img_id in common_image_ids]
    
    filtered_predictions = []
    filtered_references = []
    for pred, refs_list in zip(predictions_for_metric, references_for_metric):
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

    if bleu_metric:
        bleu_score = bleu_metric.compute(predictions=filtered_predictions, references=filtered_references)
        results['BLEU'] = bleu_score['bleu'] * 100 if 'bleu' in bleu_score else 0.0

    if rouge_metric:
        rouge_score = rouge_metric.compute(predictions=filtered_predictions, references=filtered_references)
        results['ROUGE'] = rouge_score['rougeL'] * 100 if 'rougeL' in rouge_score else 0.0

    if meteor_metric:
        meteor_score = meteor_metric.compute(predictions=filtered_predictions, references=filtered_references)
        results['METEOR'] = meteor_score['meteor'] * 100 if 'meteor' in meteor_score else 0.0

    if cider_metric:
        cider_score = cider_metric.compute(predictions=filtered_predictions, references=filtered_references)
        results['CIDEr'] = cider_score['cider'] * 100 if 'cider' in cider_score else 0.0
    
    avg_f1, avg_recall = calculate_f1_recall_token_level(predictions, references, vocab) 
    results['F1_token_avg'] = avg_f1 * 100
    results['Recall_token_avg'] = avg_recall * 100

    return results