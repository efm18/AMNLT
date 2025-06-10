from utils import levenshtein

def compute_metric(a1, a2):
    acc_ed_dist = 0
    acc_len = 0
    
    for (h, g) in zip(a1, a2):
        acc_ed_dist += levenshtein(h, g)
        acc_len += len(g)
    
    return 100.*acc_ed_dist / acc_len

def amnlt_metrics(hyp_cer, gt_cer):
    return compute_metric(hyp_cer, gt_cer)  # CER

def extract_music_text(array):
    lines = array.split("\n")
    lyrics = []
    symbols = []
    for idx, l in enumerate(lines):
        if '.\t.\n' in l:
            continue
        if idx > 0 and len(l.rstrip().split('\t')) > 1:
            symbols.append(l.rstrip().split('\t')[0])
            lyrics.append(l.rstrip().split('\t')[1])
 
    return lyrics, symbols, " ".join(lyrics)
    