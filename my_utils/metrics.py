import torch
import jiwer
import sys

# Currently not working
#from antlr4 import *
#from antlr.gabc_lexer import gabc_lexer
#from antlr.gabc_grammar import gabc_grammar
#from antlr.gabc_music_grammar import gabc_music_grammar
#from antlr4.error.ErrorListener import ErrorListener

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
    #print(y_pred)
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

# def ctc_grammar_driven_decoder(y_pred, i2w):
#     # Generate multiple possible sequences from CTC output
#     possible_sequences = simplified_beam_search(y_pred)

#     grammatically_correct_sequences = []
#     antlr_parser = None

#     # Attempt to parse each sequence
#     for sequence in possible_sequences:
#         token_string = indices_to_string(sequence, i2w)
#         input_stream = InputStream(token_string)
#         lexer = gabc_lexer(input_stream)
#         stream = CommonTokenStream(lexer)
#         parser = gabc_music_grammar(stream)
        
#         # Custom error listener
#         lexer.removeErrorListeners()
#         parser.removeErrorListeners()
#         myErrorListener = NoVerboseErrorListener()
#         lexer.addErrorListener(myErrorListener)
#         parser.addErrorListener(myErrorListener)
        
#         #print(token_string)
        
#         try:
#             antlr_parser = parser.start()
#             grammatically_correct_sequences.append(token_string)
#         except Exception as e:
#             continue

#     # Select the best sequence based on the most probable within the correct ones
#     if grammatically_correct_sequences:
#         print("Correct")
#         return grammatically_correct_sequences[0]
#     else:
#         return indices_to_string(possible_sequences[0], i2w)
    
#     #return grammatically_correct_sequences[0] if grammatically_correct_sequences else indices_to_string(possible_sequences[0], i2w)

# # Custom antlr error listener to avoid printing the errors of unrecogninez tokens by the grammar when performing the validation and test of the model
# class NoVerboseErrorListener(ErrorListener):
#     def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
#         # Overrides the default method to prevent printing syntax errors
#         pass

#     def reportAmbiguity(self, recognizer, dfa, startIndex, stopIndex, exact, ambigAlts, configs):
#         # Overrides the default method to prevent printing ambiguity errors
#         pass

#     def reportAttemptingFullContext(self, recognizer, dfa, startIndex, stopIndex, conflictingAlts, configs):
#         # Overrides the default method to prevent printing attempting full context errors
#         pass

#     def reportContextSensitivity(self, recognizer, dfa, startIndex, stopIndex, prediction, configs):
#         # Overrides the default method to prevent printing context sensitivity errors
#         pass

# # Beam search-like algorithm to retrive the 10 most possible CTCs
# def simplified_beam_search(log_probs, k=10):
#     # log_probs should be of shape [seq_len, num_classes]
#     sequences = []
#     original_log_probs = log_probs.clone()

#     for _ in range(k):
#         seq = torch.argmax(log_probs, dim=1)  # Most probable classes
#         sequences.append(seq)

#         # Mask the chosen probabilities by setting them to a very low value
#         for t, idx in enumerate(seq):
#             log_probs[t, idx] = -float('inf')  # Masking out the selected index
        
#     # Reset log_probs to original
#     log_probs = original_log_probs.clone()    

#     return sequences

# # Convert sequence of indices to a string of tokens
# def indices_to_string(sequence, i2w):
#     # Merge repeated elements
#     y_pred_decoded = torch.unique_consecutive(sequence, dim=0).tolist()
#     # Convert to string; len(i2w) -> CTC-blank
#     y_pred_decoded = [i2w[i] for i in y_pred_decoded if i != len(i2w)]
#     return y_pred_decoded