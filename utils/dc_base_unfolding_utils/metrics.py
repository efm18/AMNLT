import torch
import jiwer
import sys

# -------------------------------------------- METRICS:


def levenshtein(a, b):
    n, m = len(a), len(b)

    if n > m:
        a, b = b, a
        n, m = m, n

    current = range(n + 1)
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]


def compute_cer(y_true, y_pred):
    ed_acc = 0
    length_acc = 0
    for t, h in zip(y_true, y_pred):
        ed_acc += levenshtein(t, h)
        length_acc += len(t)
    return 100.0 * ed_acc / length_acc


def compute_metrics(y_true, y_pred, music):
    #metrics = {"cer": compute_cer(y_true, y_pred)}
    
    if music:
        y_true = ["".join(s).replace("(", "").replace(")", "") for s in y_true]
        y_pred = ["".join(s).replace("(", "").replace(")", "") for s in y_pred]

    else:
        y_true = ["".join(s) for s in y_true]
        y_pred = ["".join(s) for s in y_pred]
        
    metrics = {"cer": jiwer.cer(y_true, y_pred) * 100.00}
    #metrics = {"wer": jiwer.wer(y_true, y_pred) * 100.00}
    return metrics


# -------------------------------------------- CTC DECODERS:

def ctc_greedy_decoder(y_pred, i2w):
    # y_pred = [seq_len, num_classes]
    # Best path
    y_pred_decoded = torch.argmax(y_pred, dim=1)
    # Save ctc frames for post-allignment
    with open('ctc_predictions.txt', 'a') as f:
        f.write(f"{''.join([i2w[i] if i != len(i2w) else 'ñ' for i in y_pred_decoded.tolist()])}\n")
    # Merge repeated elements
    y_pred_decoded = torch.unique_consecutive(y_pred_decoded, dim=0).tolist()
    # Convert to string; len(i2w) -> CTC-blank
    y_pred_decoded = [i2w[i] for i in y_pred_decoded if i != len(i2w)]
    return y_pred_decoded