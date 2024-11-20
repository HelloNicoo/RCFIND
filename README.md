# <p align="center"> Reading Comprehension Powered Semantic Fusion Network for Identification of N-ary Drug Combinations</p>

## ✨ Introduction
This repository contains the code and data for the paper titled "Reading Comprehension Powered Semantic Fusion Network for Identification of N-ary Drug Combinations".This paper introduces a machine reading comprehension-driven semantic fusion network for identifying N-ary drug combinations.This README provides an overview of the repository and instructions for running the code and using the data.


## 📃 Data


The NDCI dataset is originally sourced from  [here][4]. 

Our team has processed this data and made it available for your convenience at [Google Drive][1].

[1]: https://drive.google.com/file/d/1PTCvLFV0rX7cFKtmCZ-T8xY2MXBNlQDC/view?usp=sharing
[4]: https://github.com/allenai/drug-combo-extraction


## 🚀 Quick Start
# ⚙️ Setup
To run the code in this repository, you'll need the following dependencies:

&bull; Python 3.9

&bull; PyTorch 2.1

&bull; transformers

Install these dependencies using pip:
```
conda create -n Drug-MRC python=3.9
conda activate Drug-MRC
pip install -r requirements.txt
```

🤖 Download Pre-trained Model

Before executing the code, you need to download the pre-trained model [PubMedBert][2].

[2]: https://huggingface.co/spaces/MarfarsLi/microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext

⚡️ Running the Code

&bull; Model Training:

```
python main.py --do_train
```

&bull; Model Evaling:

We release the RCFIND  model (one seed): model.pkl [Google Drive][3]. You can run it with the following command:
```
python main.py --do_eval
```

[3]: https://drive.google.com/file/d/1PTCvLFV0rX7cFKtmCZ-T8xY2MXBNlQDC/view?usp=sharing
