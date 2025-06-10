from collections import Counter
import os
import sys
import jiwer
import re
import json

# Set to True to save per-sample metrics in JSON
save_per_sample_metrics = True
output_json = "per_sample_metrics.json"  # Output filename

def get_bwer(X, Y, verbose=False):
    fxv = Counter(X)
    fyv = Counter(Y)
    V = set(fxv.keys()) | set(fyv.keys())
    diff_count = 0
    for word in V:
        diff_count += abs(fxv[word] - fyv[word])
    if verbose:
        print(f"Diff count: {diff_count}")
        print(f"Length of X: {len(X)}")
        print(f"Numerator: {abs(len(X) - len(Y))}")
    diff_count = (abs(len(X) - len(Y)) + diff_count) / (2 * len(X))
    return diff_count * 100.00

def get_mer(gt_files, resultado_final, dataset, verbose=False):
    total_mer = 0.0
    mer_values = []
    random_sample = False
    worst_mer = 0.0
    for gt_file, result in zip(gt_files, resultado_final):
        base_name, _ = os.path.splitext(gt_file)
        if dataset == "solesmes":
            gt_file = "../data/Repertorium_GT_music/" + base_name + ".gabc"
        elif dataset == "gregosynth":
            gt_file = "../data/GT_music/" + base_name + ".gabc"
        elif dataset == "einsiedeln":
            gt_file = "../data/Einsiedeln_GT_music/" + base_name + ".gabc"
        elif dataset == "salzinnes":
            gt_file = "../data/Salzinnes_GT_music_v2/" + base_name + ".gabc"
        with open(gt_file.strip(), 'r', encoding='utf-8') as gt_file:
            gt_text = gt_file.read().replace('\n', ' ')
        gt_text = gt_text.replace('(', ' ').replace(')', ' ')
        gt_text = ' '.join(gt_text.split())
        result = ' '.join(result.split())
        if dataset in ["solesmes", "gregosynth"]:
            mer = jiwer.cer(gt_text, result)
        elif dataset in ["einsiedeln", "salzinnes"]:
            mer = jiwer.wer(gt_text, result)
        total_mer += mer
        mer_values.append(mer * 100.00)
        
        if mer > worst_mer:
            worst_mer = mer
            worst_gt = gt_text
            worst_result = result
        if random_sample:
            print(gt_text)
            print(result)
            print(mer)
            random_sample = False
            
        if base_name == "034r_1":
            print(mer * 100.00)
    if verbose:
        print("Worst MER:" + str(worst_mer))
        print("\tGT: " + str(worst_gt))
        print("\tPrediction: " + str(worst_result) + "\n")
    mean_mer = (total_mer / len(gt_files)) * 100.00
    
    # Calculate standard deviation
    mean_mer2 = sum(mer_values) / len(mer_values)
    variance = sum((x - mean_mer2) ** 2 for x in mer_values) / len(mer_values)
    std_dev = variance ** 0.5
    print(f"Mean MER: {mean_mer2:.2f}")
    print(f"Standard Deviation: {std_dev:.2f}")
    print(f"Minimum MER: {min(mer_values):.2f}")
    print(f"Maximum MER: {max(mer_values):.2f}")
    print(f"Median MER: {sorted(mer_values)[len(mer_values) // 2]:.2f}")
    print(f"Variance: {variance:.2f}")
    
    return mean_mer

def get_cer_syler(gt_files, resultado_final, dataset, verbose=False):
    total_cer = 0.0
    total_syler = 0.0
    cer_values = []
    random_sample = False
    worst_cer = 0.0
    for gt_file, result in zip(gt_files, resultado_final):
        base_name, _ = os.path.splitext(gt_file)
        if dataset == "solesmes":
            gt_file = "../data/Repertorium_GT_lyrics/" + base_name + ".gabc"
        elif dataset == "gregosynth":
            gt_file = "../data/GT_lyrics/" + base_name + ".gabc"
        elif dataset == "einsiedeln":
            gt_file = "../data/Einsiedeln_GT_lyrics/" + base_name + ".gabc"
        elif dataset == "salzinnes":
            gt_file = "../data/Salzinnes_GT_lyrics_v2/" + base_name + ".gabc"
        with open(gt_file.strip(), 'r', encoding='utf-8') as gt_file:
            gt_text = gt_file.read().replace('\n', ' ')
        gt_text = ' '.join(gt_text.split())
        result = ' '.join(result.split())
        cer = jiwer.cer(gt_text, result)
        syler = jiwer.wer(gt_text, result)
        total_cer += cer
        total_syler += syler
        cer_values.append(cer * 100.00)
        if random_sample:
            print(gt_text)
            print(result)
            print(cer)
            random_sample = False
        if cer > worst_cer:
            worst_cer = cer
            worst_gt = gt_text
            worst_result = result
        if base_name == "034r_1":
            print(cer * 100.00)
            print(syler * 100.00)
    if verbose:
        print("Worst CER:" + str(worst_cer))
        print("\tGT: " + str(worst_gt))
        print("\tPrediction: " + str(worst_result) + "\n")
    mean_cer = (total_cer / len(gt_files)) * 100.00
    mean_syler = (total_syler / len(gt_files)) * 100.00
    return mean_cer, mean_syler

def extract_music_lyrics_solesmes_gregosynth_muaw(archivo_transcripcion, gt_file, dataset):
    with open(archivo_transcripcion, 'r', encoding='utf-8') as transcripts_file:
        transcripts_lines = transcripts_file.readlines()
    resultado_final_music = []
    resultado_final_lyrics = []
    for transcript in transcripts_lines:
        with_m_string = ""
        without_m_string = ""
        i = 0
        while i < len(transcript):
            if transcript[i:i+3] == "<m>":
                with_m_string += transcript[i+3]
                without_m_string += ' '
                i += 4
            else:
                if transcript[i] != ")" and transcript[i] != "(":
                    without_m_string += transcript[i]
                with_m_string += ' '
                if transcript[i] != ' ':
                    i += 1
                else:
                    without_m_string += ' '
                    i += 1
        with_m_string = re.sub(r'\s+', ' ', with_m_string)
        without_m_string = re.sub(r'\s+', ' ', without_m_string)
        resultado_final_music.append(with_m_string)
        resultado_final_lyrics.append(without_m_string)
    with open(gt_file, 'r', encoding='utf-8') as gt_file:
        gt_files = gt_file.readlines()
    mer = get_mer(gt_files, resultado_final_music, dataset)
    cer, syler = get_cer_syler(gt_files, resultado_final_lyrics, dataset)
    return mer, cer, syler

def extract_music_lyrics_einsiedeln_salzinnes(archivo_transcripcion, gt_file, dataset):
    with open(archivo_transcripcion, 'r', encoding='utf-8') as transcripts_file:
        transcripts_lines = transcripts_file.readlines()
    resultado_final_music = []
    resultado_final_lyrics = []
    if dataset == "einsiedeln":
        music_vocab = "../data/einsiedeln_music/vocab/w2i_new_gabc.json"
    elif dataset == "salzinnes":
        music_vocab = "../data/salzinnes_music/vocab/w2i_new_gabc.json"
    with open(music_vocab, 'r', encoding='utf-8') as vocab_file:
        music_vocab = json.load(vocab_file)
        sorted_music_vocab = sorted(music_vocab.keys(), key=len, reverse=True)
    for transcript in transcripts_lines:
        found_music_tokens = []
        transcript_copy = transcript
        for token in sorted_music_vocab:
            if token in transcript_copy:
                if token in ["n", "f"]:
                    pattern = r'(?<!\()\b{}\b(?!\))'.format(re.escape(token))
                    matches = re.findall(pattern, transcript_copy)
                    if matches:
                        found_music_tokens.append(token)
                        transcript_copy = re.sub(pattern, " ", transcript_copy)
                else:
                    found_music_tokens.append(token)
                    transcript_copy = transcript_copy.replace(token, " ")
        resultado_final_lyrics.append(" ".join(transcript_copy.split()))
        i = 0
        transcript_music = []
        while i < len(transcript):
            for token in found_music_tokens:
                if transcript[i:i+len(token)] == token:
                    transcript_music.append(token)
                    i += len(token)
                    break
            else:
                i += 1
        resultado_final_music.append(" ".join(transcript_music).replace("(", "").replace(")", ""))
    with open(gt_file, 'r', encoding='utf-8') as gt_file:
        gt_files = gt_file.readlines()
    mer = get_mer(gt_files, resultado_final_music, dataset)
    cer, syler = get_cer_syler(gt_files, resultado_final_lyrics, dataset)
    return mer, cer, syler

def tokenization_muaw(X, Y):
    words_X = []
    music = False
    syl = ""
    for char in X:
        if char == "(":
            if syl != "":
                words_X.append(syl)
                syl = ""
            words_X.append(char)
            music = True
        elif char == ")":
            words_X.append(char)
            music = False
        elif music:
            words_X.append(char)
        elif not music and char != " ":
            syl += char
    words_Y = []
    i = 0
    syl = ""
    while i < len(Y):
        if Y[i] == "(":
            if(syl != ""):
                words_Y.append(syl)
                syl = ""
            words_Y.append(Y[i])
        elif Y[i] == ")":
            words_Y.append(Y[i])
        elif Y[i:i+3] == "<m>":
            words_Y.append(Y[i+3])
            i += 3
        elif Y[i] != " ":
            syl += Y[i]
        i += 1
    return words_X, words_Y

def tokenization_pseudo(X, Y):
    if dataset == "einsiedeln":
        music_vocab = "../data/einsiedeln_music/vocab/w2i_new_gabc.json"
    elif dataset == "salzinnes":
        music_vocab = "../data/salzinnes_music/vocab/w2i_new_gabc.json"
    with open(music_vocab, 'r', encoding='utf-8') as vocab_file:
        music_vocab = json.load(vocab_file)
        sorted_music_vocab = sorted(music_vocab.keys(), key=len, reverse=True)
    X = X.replace(" ", "")
    Y = Y.replace(" ", "")
    words_X = extract_vocab(sorted_music_vocab, X)
    words_Y = extract_vocab(sorted_music_vocab, Y)
    return words_X, words_Y

def extract_vocab(sorted_music_vocab, transcript):
    lyrics_vocab = []
    found_music_tokens = []
    words = []
    transcript_copy = transcript
    for token in sorted_music_vocab:
        if token in transcript_copy:
            found_music_tokens.append(token)
            transcript_copy = transcript_copy.replace(token, " ")
    lyrics_vocab.append(" ".join(transcript_copy.split()))
    vocab = found_music_tokens + transcript_copy.split()
    sorted_vocab = sorted(vocab, key=len, reverse=True)
    i = 0
    while i < len(transcript):
        for token in sorted_vocab:
            if transcript[i:i+len(token)] == token:
                words.append(token)
                i += len(token)
                break
        else:
            i += 1
    return words

def metrics(transcript, gt_file, dataset, n_samples):
    if dataset in ["solesmes", "gregosynth"]:
        mer, cer, syler = extract_music_lyrics_solesmes_gregosynth_muaw(transcript, gt_file, dataset)
    elif dataset in ["einsiedeln", "salzinnes"]:
        mer, cer, syler = extract_music_lyrics_einsiedeln_salzinnes(transcript, gt_file, dataset)
    with open(transcript, 'r', encoding='utf-8') as transcript_file:
        transcript_lines = transcript_file.readlines()
    with open(gt_file, 'r', encoding='utf-8') as gt_file:
        gt_files = gt_file.readlines()
    gt_texts = []
    for gt_file in gt_files:
        base_name, _ = os.path.splitext(gt_file)
        if dataset == "solesmes":
            gt_file_path = "../data/Repertorium_GT_full/" + base_name + ".gabc"
        elif dataset == "gregosynth":
            gt_file_path = "../data/GT/" + base_name + ".gabc"
        elif dataset == "einsiedeln":
            gt_file_path = "../data/Einsiedeln_GT/" + base_name + ".gabc"
        elif dataset == "salzinnes":
            gt_file_path = "../data/Salzinnes_GT_v2/" + base_name + ".gabc"
        with open(gt_file_path.strip(), 'r', encoding='utf-8') as gt_file_inner:
            gt_file = gt_file_inner.read().replace('\n', ' ')
            gt_texts.append(gt_file)
    y_true = ["".join(s) for s in gt_texts]
    y_pred = ["".join(s) for s in transcript_lines]
    per_sample = []
    N = len(y_true)
    if n_samples == 0 or n_samples > N:
        n_samples = N
    for i in range(n_samples):
        if dataset in ["solesmes", "gregosynth"]:
            x_tokens, y_tokens = tokenization_muaw(gt_texts[i], transcript_lines[i])
        elif dataset in ["einsiedeln", "salzinnes"]:
            x_tokens, y_tokens = tokenization_pseudo(gt_texts[i], transcript_lines[i])
        sample_bwer = get_bwer(x_tokens, y_tokens)
        sample_amler = jiwer.wer(" ".join(x_tokens), " ".join(y_tokens)) * 100.00
        sample_aler = (sample_amler - sample_bwer) / sample_amler if sample_amler > 0 else 0
        per_sample.append({
            "file": gt_files[i].strip(),
            "gt": gt_texts[i],
            "pred": transcript_lines[i],
            "bwer": sample_bwer,
            "amler": sample_amler,
            "aler": sample_aler
        })
    if save_per_sample_metrics:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(per_sample, f, indent=2, ensure_ascii=False)
        print(f"Saved per-sample metrics in {output_json}")
    total_bwer = 0
    total_amler = 0
    for x, y in zip(y_true, y_pred):
        if dataset in ["solesmes", "gregosynth"]:
            x_tokens, y_tokens = tokenization_muaw(x, y)
        elif dataset in ["einsiedeln", "salzinnes"]:
            x_tokens, y_tokens = tokenization_pseudo(x, y)
        total_bwer += get_bwer(x_tokens, y_tokens)
        total_amler += jiwer.wer(" ".join(x_tokens), " ".join(y_tokens)) * 100.00
    bwer = total_bwer / len(y_true)
    amler = total_amler / len(y_true)
    aler = (amler - bwer) / amler if amler > 0 else 0
    print(f"MER: {round(mer, 2)}")
    print(f"CER: {round(cer, 2)}")
    print(f"SylER: {round(syler, 2)}")
    print(f"ALER: {round(aler, 2)}")
    print(f"\tAMLER: {round(amler, 2)}")
    print(f"\tBWER: {round(bwer, 2)}")

if __name__ == "__main__":
    test = False
    dataset = "einsiedeln"
    model = "dan"
    dc = False
    alignment = "tm_grouped"
    n_samples = 0  # set to 0 for all, or any number for partial output

    if dc:
        if dataset == "salzinnes":
            transcript_file = f"salzinnes/predictions_salzinnes_{alignment}_aligned.txt"
            gt_file = "../data/Salzinnes_Folds/test_gt_fold.dat"
        elif dataset == "einsiedeln":
            transcript_file = f"einsiedeln/predictions_einsiedeln_{alignment}_aligned.txt"
            gt_file = "../data/Einsiedeln_Folds/test_gt_fold.dat"
        elif dataset == "gregosynth":
            transcript_file = f"gregosynth/predictions_gregosynth_{alignment}_aligned.txt"
            gt_file = "../data/Folds/test_gt_fold.dat"
        elif dataset == "solesmes":
            transcript_file = f"solesmes/predictions_solesmes_{alignment}_aligned.txt"
            gt_file = "../data/Repertorium_Folds/test_gt_fold.dat"
    else:
        if dataset == "solesmes":
            transcript_file = "solesmes/predictions_solesmes_" + model + "_music_aware.txt"
            gt_file = "../data/Repertorium_Folds/test_gt_fold.dat"
        elif dataset == "gregosynth":
            transcript_file = "gregosynth/predictions_gregosynth_" + model + "_music_aware.txt"
            gt_file = "../data/Folds/test_gt_fold.dat"
        elif dataset == "einsiedeln":
            transcript_file = "einsiedeln/predictions_einsiedeln_" + model + ".txt"
            gt_file = "../data/Einsiedeln_Folds/test_gt_fold.dat"
        elif dataset == "salzinnes":
            transcript_file = "salzinnes/predictions_salzinnes_" + model + ".txt"
            gt_file = "../data/Salzinnes_Folds/test_gt_fold.dat"

    metrics(transcript_file, gt_file, dataset, n_samples)
