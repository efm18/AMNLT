<p align="center">
  <a href=""><img src="https://i.imgur.com/Iu7CvC1.png" alt="PRAIG-logo" width="100"></a>
</p>

<h1 align="center">Aligned Music Notation and Lyrics Transcription</h1>

<h4 align="center">📄 Full paper available <a href="" target="_blank">here</a>.</h4>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9.0-orange" alt="Gitter">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white" alt="Lightning">
  <img src="https://img.shields.io/static/v1?label=License&message=MIT&color=blue" alt="License">
</p>


<p align="center">
  <a href="#about">About</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#citations">Citations</a> •
  <a href="#acknowledgments">Acknowledgments</a> •
  <a href="#license">License</a>
</p>

<a name=about></a>
## 📚 About

This repository contains the official code for Aligned Music Notation and Lyrics Transcription (AMNLT), a framework designed to jointly transcribe and align music notation and lyrics from vocal score images—especially historical notations such as Gregorian chant.

Traditional OMR (Optical Music Recognition) and OCR (Optical Character Recognition) systems typically treat music and lyrics independently, failing to preserve their crucial alignment. AMNLT introduces a comprehensive solution by integrating transcription and alignment into a unified task, enabling meaningful digital representations of vocal music.

This project includes:
- Divide-and-conquer and end-to-end models for AMNLT.
- Post-processing alignment strategies (syllable-level, frame-level).
- Holistic transcription pipelines (CTC, unfolding, and language modeling).
- Benchmark datasets (GregoSynth, Solesmes, Einsiedeln, Salzinnes) uploaded to [Hugging Face](https://huggingface.co/datasets/PRAIG/AMNLT).
- Metrics for transcription accuracy and alignment precision (MER, CER, SylER, AMLER, AlER).

<a name=how-to-use></a>
## ⚙️ How to Use

To run the code, you'll need to meet certain requirements which are specified in the [`Dockerfile`](Dockerfile). Alternatively, you can set up a virtual environment if preferred.

Once you have prepared your environment (either a Docker container or a virtual environment), you are ready to begin. Execute the [`main.py`](main.py) script to replicate the experiments from our work:

```python
python main.py [arguments]
```
> ⚠️ **Note**  
> This script requires one or more arguments. Run python `main.py --help` to see the full list of options and usage instructions.

<a name=citations></a>
## 📖 Citations

```bibtex
@article{john2025doe,
  title     = {{}},
  author    = {},
  journal   = {{}},
  volume    = {},
  pages     = {},
  year      = {},
  publisher = {},
  doi       = {},
}
```

<a name=acknowledgments></a>
## 🙏 Acknowledgments

This work is part of the [REPERTORIUM](https://repertorium.eu/) project, funded by EU Horizon Europe program (Grant No. 101095065).

<a name=license></a>
## 📝 License

This work is under a [MIT](https://opensource.org/licenses/MIT) license.

> ⚠️ **Disclaimer**  
> This project is currently undergoing a major refactoring. Some parts of the codebase may not work as expected or could change without notice.  
> Please proceed with caution and check back for updates.
