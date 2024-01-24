# Knowledge Generation for Zero-shot Knowledge-based VQA

This includes an original implementation of "[Knowledge Generation for Zero-shot Knowledge-based VQA][paper]" by Rui Cao, Jing Jiang.

<p align="center">
  <img src="kgen_arch.PNG" width="80%" height="80%">
</p>

This code provides:
- Codes for generating relevant knowledge for knowlege-intensive VQA questions with GPT-3.
- Generated knowledge from GPT-3 for each K-VQA dataset
- Codes to incorporate the generated knowledge into K-VQA models based on 1) UnifiedQA, 2) OPT and 3) GPT-3

Please leave issues for any questions about the paper or the code.

If you find our code or paper useful, please cite the paper:
```
@inproceedings{ caojiang2024kgenvqa,
    title={Knowledge Generation for Zero-shot Knowledge-based VQA},
    author={ Rui Cao, Jing Jiang},
    journal={EACL},
    year={ 2024 }
}
```

### Announcements
01/18/2023: Our paper is accepted by EACL, 2024, as Findings. 

## Content
1. [Installation](#installation)
2. [Prepare Datasets](#prepare-datasets)
    * [Step 1: Downloading Datasets](#step-1-down-dataset)
    * [Step 2: Caption Generation](#step-2-cap-gen) 
4. [Knowledge Generation](#kb-generation) (Section 3.1 of the paper)
    * [Step 1: Knowledge Initialization](#step-1-kb-init) 
    * [Step 2: Knowledge Diversification](#step-2-kb-diverse) 
5. [Experiments: Incorporating Generated Knowledge for K-VQA](#experiments) (Section 3.2 and 4.3 of the paper)
    * [K-VQA based on UnifiedQA](#performance-uniqa)
    * [K-VQA based on OPT](#prompthate-opt)
    * [K-VQA based on GPT](#prompthate-gpt)   

## Installation
The code is tested with python 3.8. To run the code, you should install the package of transformers provided by Huggingface (version 4.29.2), PyTorch library (1.13.1 version), LAVIS package from Salesforce (version 1.0.2). The code is implemented with the CUDA of 11.2 (you can also implement with other compatible versions) on Tesla V 100 GPU card (each with 32G dedicated memory). Besides running the OPT model, all other models take one GPU each.

###
## Prepare Datasets  

### Step 1: Downloading Datasets 
We have tested on three benchmarks for knowledge-intensive VQA (K-VQA) datasets: *OK-VQA* and *A-OKVQA*. Datasets are available online. You can download datasets via links in the original dataset papers and put them into the desired file paths according to the code: OK_PATH, A_OK_PATH, PATH.

### Step 2: Caption Generation  
For memes, we conduct data pre-processing

## Knowledge Generation

Here we describe how we generate Pro-Cap with frozen pre-trained vision-language models (PT-VLMs). Specifically, we design several probing questions highly related to hateful content detection and prompt PT-VLMs with these questions and meme images. 

### Step 1: Knowledge Initialization
To alleviate noisy in input images when prompting PT-VLMs, we detect meme texts on images, remove texts and conduct image impaintings to obtain *clean images*. Due to the data privacy policy, we are unable to share the cleaned images here. Specifically, we use EasyOCR for text detection and MMEditing for image impainting. 

### Step 2: Knowledge Diversification
We next prompt frozen PT-VLMs with questions and cleaned images to obtain Pro-Cap. You can generate Pro-Cap with our code at [codes/Pro-Cap-Generation.ipynb](codes/Pro-Cap-Generation.ipynb). Or you can alternatively use generated Pro-Cap shared in [codes/Ask-Captions](codes/Ask-Captions).


### Experiments: Incorporating Generated Knowledge for K-VQA
Before uploading codes, we re-run the codes. Because of the updating of the versions of transformers package, we observe a small variance compared with the reported performance in the paper. We conclude both the reported results and the re-implemented result in the Figure above. There is no significant difference according to p-value. We share both the re-implemented logger files and the logger files for the reported performance in [codes/logger](codes/logger) and [codes/reported](codes/reporte).

### K-VQA based on UnifiedQA  
place holder   

### K-VQA based on OPT  
place holder   

### K-VQA based on GPT  
place holder   



[paper]: https://arxiv.org/abs/2308.08088
