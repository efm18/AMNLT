DS_CONFIG = {
    # Baseline CRNN + post-alignment
    "gregoeli_lyrics_croppedImage": {
        "train": "data/Folds/train_gt_fold.dat",
        "val": "data/Folds/val_gt_fold.dat",
        "test": "data/Folds/test_gt_fold.dat",
        "transcripts": "data/GT_lyrics",
        "images": "data/Images_lyrics",
    },
    "gregoeli_lyrics_fullImage": {
        "train": "data/Folds/train_gt_fold.dat",
        "val": "data/Folds/val_gt_fold.dat",
        "test": "data/Folds/test_gt_fold.dat",
        "transcripts": "data/GT_lyrics",
        "images": "data/Images",
    },
    "gregoeli_music_croppedImage": {
        "train": "data/Folds/train_gt_fold.dat",
        "val": "data/Folds/val_gt_fold.dat",
        "test": "data/Folds/test_gt_fold.dat",
        "transcripts": "data/GT_music",
        "images": "data/Images_music",
    },
    "gregoeli_music_fullImage": {
        "train": "data/Folds/train_gt_fold.dat",
        "val": "data/Folds/val_gt_fold.dat",
        "test": "data/Folds/test_gt_fold.dat",
        "transcripts": "data/GT_music",
        "images": "data/Images",
    },
    # Repertorium
    "repertorium_music": {
        "train": "data/Repertorium_Folds/train_gt_fold.dat",
        "val": "data/Repertorium_Folds/val_gt_fold.dat",
        "test": "data/Repertorium_Folds/test_gt_fold.dat",
        "transcripts": "data/Repertorium_GT_music",
        "images": "data/Repertorium_Images_music"
    },
    "repertorium_lyrics": {
        "train": "data/Repertorium_Folds/train_gt_fold.dat",
        "val": "data/Repertorium_Folds/val_gt_fold.dat",
        "test": "data/Repertorium_Folds/test_gt_fold.dat",
        "transcripts": "data/Repertorium_GT_lyrics",
        "images": "data/Repertorium_Images_lyrics"
    },
    "repertorium_full": {
        "train": "data/Repertorium_Folds/train_gt_fold.dat",
        "val": "data/Repertorium_Folds/val_gt_fold.dat",
        "test": "data/Repertorium_Folds/test_gt_fold.dat",
        "transcripts": "data/Repertorium_GT_full",
        "images": "data/Repertorium_Images_full"
    },
    "repertorium_full_music_aware": {
        "train": "data/Repertorium_Folds/train_gt_fold.dat",
        "val": "data/Repertorium_Folds/val_gt_fold.dat",
        "test": "data/Repertorium_Folds/test_gt_fold.dat",
        "transcripts": "data/Repertorium_GT_music_aware",
        "images": "data/Repertorium_Images_full"
    },
    # Baseline FCN
    "gregoeli_fullImage": {
        "train": "data/Folds/train_gt_fold.dat",
        "val": "data/Folds/val_gt_fold.dat",
        "test": "data/Folds/test_gt_fold.dat",
        "transcripts": "data/GT",
        "images": "data/Images",
    },
    # Music aware encoding FCN
    "gregoeli_fullImage_music_aware": {
        "train": "data/Folds/train_gt_fold.dat",
        "val": "data/Folds/val_gt_fold.dat",
        "test": "data/Folds/test_gt_fold.dat",
        "transcripts": "data/GT_music_aware",
        "images": "data/Images",
    },
    # Einsiedeln
    "einsiedeln": {
        "train": "data/Einsiedeln_Folds/train_gt_fold.dat",
        "val": "data/Einsiedeln_Folds/val_gt_fold.dat",
        "test": "data/Einsiedeln_Folds/test_gt_fold.dat",
        "transcripts": "data/Einsiedeln_GT",
        "images": "data/Einsiedeln_Images",
    },
    "einsiedeln_music": {
        "train": "data/Einsiedeln_Folds/train_gt_fold.dat",
        "val": "data/Einsiedeln_Folds/val_gt_fold.dat",
        "test": "data/Einsiedeln_Folds/test_gt_fold.dat",
        "transcripts": "data/Einsiedeln_GT_music",
        "images": "data/Einsiedeln_Images_music",
    },
    "einsiedeln_lyrics": {
        "train": "data/Einsiedeln_Folds/train_gt_fold.dat",
        "val": "data/Einsiedeln_Folds/val_gt_fold.dat",
        "test": "data/Einsiedeln_Folds/test_gt_fold.dat",
        "transcripts": "data/Einsiedeln_GT_lyrics",
        "images": "data/Einsiedeln_Images_lyrics",
    },
    #Salzinnes
    "salzinnes": {
        "train": "data/Salzinnes_Folds/train_gt_fold.dat",
        "val": "data/Salzinnes_Folds/val_gt_fold.dat",
        "test": "data/Salzinnes_Folds/test_gt_fold.dat",
        "transcripts": "data/Salzinnes_GT_v2",
        "images": "data/Salzinnes_Images_v2",
    },
    "salzinnes_music": {
        "train": "data/Salzinnes_Folds/train_gt_fold.dat",
        "val": "data/Salzinnes_Folds/val_gt_fold.dat",
        "test": "data/Salzinnes_Folds/test_gt_fold.dat",
        "transcripts": "data/Salzinnes_GT_music_v2",
        "images": "data/Salzinnes_Images_music_v2",
    },
    "salzinnes_lyrics": {
        "train": "data/Salzinnes_Folds/train_gt_fold.dat",
        "val": "data/Salzinnes_Folds/val_gt_fold.dat",
        "test": "data/Salzinnes_Folds/test_gt_fold.dat",
        "transcripts": "data/Salzinnes_GT_lyrics_v2",
        "images": "data/Salzinnes_Images_lyrics_v2",
    }
}