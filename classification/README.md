# Scanned document classification: 
1. **Image classification with inter and intra domain**: domain_classification.ipynb
2. **Multi-modal Transformer model**: multi-modal Transformer_layoutlm.ipynb
3. **Text extraction + text classification**: textClassifierHATT.py 


## Solution 1: Image classification with inter and intra domain

**Source**: https://github.com/arpan65/Scanned-document-classification-using-deep-learning

**Data**: [RVL-CDIP (Ryerson Vision Lab Complex Document Information Processing) data set](https://www.cs.cmu.edu/~aharley/rvl-cdip/) which consists of 400,000 grayscale images in 16 classes, with 25,000 images per class. There are 320,000 training images, 40,000 validation images, and 40,000 test images. The images are sized so their largest dimension does not exceed 1000 pixels. The size of this data set is more than 200 GB. Address is rvl-cdip.tar

**Method**:
1. Import image data with ImageDataGenerator (from tf.keras.preprocessing.image): N_generated_img = number of epochs * batch size * steps_per_epoch with the standard given in the method. Data augmentation is optional here due to little change to result.
2. Uses inter and intra domain transfer learning where an image is divided into four parts header, footer, left body and right body. 
  - Common: A pretrained VGG16 model is first used to train over the whole images(inter domain) then this model is used to train the part of images(Intra domain).
  - Here: Instead of intra domain transfer learning using VGG16, we trained two parallel models VGG16 and InceptionResNetV2 and used a stack of these as our final model. Our assumption was that because of the different architectures of these two models they will learn the different aspect of images and stacking them will result in good generalization.
3. Tuning hyperparameters: For any CNN the hyper parameters are: learning rate, pooling size, network size, batch size, choice of optimizer, regularization, input size etc.
  - learning rate: here use  ‘Cyclic Learning Rate’, which aims to train neural network such a way that the learning rate changes in a cyclic way for each training batch. It varies the learning rate within a threshold. The periodic higher learning rate helps to overcome if it stuck in the saddle point or local minima.

**Evaluation**: Accuracy, micro average F1 score, confusion metric for all classes in heatmap (See repo). Training accuracy is 97% and test accuracy is 91.45%.

why not OCR to extract text and apply NLP techniques: _Low quality scans resulted in a poor quality of text extraction_. In the practical business scenarios also we do not have control over the quality of scans, so models rely on OCR may suffer from poor generalization even after proper preprocessing.





## Solution 2: Multi-modal Transformer model 

**Source**: [LayoutLM publication](https://arxiv.org/pdf/2012.14740v1.pdf) uses a Multimodal (text + layout/format + image) pre-training for document AI. Need OCR before to extract text information. Use Pytorch ([original code](https://github.com/microsoft/unilm/tree/master/layoutlm) from Microsoft Research Team)

**Data**: 
+ [FUNSD (Form Understanding in Noisy Scanned Documents)](https://guillaumejaume.github.io/FUNSD/download/) 199 fully annotated _forms_ with 31485 words，including semantic entities and relations. The official OCR annotation is directly used with the layout information
+ [CORD (A Consolidated Receipt Dataset for Post-OCR Parsing)](https://github.com/clovaai/cord) over 11,000 Indonesian _receipts_ collected from shops and restaurants with full annotation. There are five superclass and 42 subclass labels
+ [SROIE]
+ [Kleister-NDA]
+ [DocVQA]
+ [RVL-CDIP (Ryerson Vision Lab Complex Document Information Processing) data set](https://www.cs.cmu.edu/~aharley/rvl-cdip/) as before

**Method**:

Step 1. Embedding:

(A) Text Embedding: embed the text extracted from doc
The final text embedding is the sum of three embeddings: i-th text embedding **ti = TokEmb(wi) + PosEmb1D(i) + SegEmb(si)**, 0 ≤ i < L
1) Token embedding: recognize text and serialize it in a reasonable reading order using off-theshelf OCR tools and PDF parsers, then use WordPiece to tokenize the text sequence and assign each token to a certain segment. Then add a [CLS] at the beginning of the token sequence and a [SEP] at the end of each text segment. The length of the text sequence is limited to ensure that the length of the final sequence is not greater than the maximum sequence length L. Extra [PAD] tokens are appended after the last [SEP] token to fill the gap if the token sequence is still shorter than L tokens. In this way, we get the input token sequence like _S = {[CLS], w1, w2, ..., [SEP], [PAD], [PAD], ...}, |S| = L_
2) 1D positional embedding: the token index
3) Segment embedding is used to distinguish different text segments. 

(B) Visual Embedding: embed the scanned doc
The final text embedding is the sum of three embeddings: i-th visual embedding is **vi = Proj(VisTokEmb(I)i) + PosEmb1D(i) + SegEmb([C])**, 0 ≤ i < W
1) Token: a document page image I --> resized to 224 × 224 --> visual backbone of ResNeXt-FPN to generate output feature map --> average-pooling to width W * height H --> flattened into a visual embedding sequence of length WH --> linear projection layer applied to each visual token embedding in order to unify the dimensions. 
2) ID positional embedding: Since the CNN-based visual backbone cannot capture the positional information, we also add a 1D positional embedding to these image token embeddings. The 1D positional embedding is shared with the text embedding layer. 
3) Segment embedding, we attach all visual tokens to the visual segment [C]. 

(C) Layout Embedding: embed the spatial layout information represented by token bounding boxes
Normalize and discretize all coordinates to integers in the range [0, 1000], and use two embedding layers to embed x-axis features and y-axis features separately --> normalized bounding box of the i-th text/visual token is boxi = (x0, x1, y0, y1, w, h) --> the layout embedding layer: concatenated six bounding box features **li = Concat(PosEmb2Dx(x0, x1, w),PosEmb2Dy(y0, y1, h))**, 0 ≤ i < W H + L.
(Note for dummy CLS, SEP and PAD, box = (0, 0, 0, 0, 0, 0))

Step 2. Multi-modal Encoder with Spatial-Aware Self-Attention Mechanism
1) First layer of encoder: concatenates visual embeddings {v0, ..., vWH−1} and text embeddings {t0, ..., tL−1} to a unified sequence X and fuses spatial information by adding the layout embeddings to get the first layer input x(0): xi(0) = Xi + li, where X = {v0, ..., vWH−1, t0, ..., tL−1}
2) Spatial-aware self-attention mechanism into the self-attention layers


**Evaluation**: 
Accuracy among different datasets, benchmarked with other commonly used models. forms not as good as receipt or varied doc.


## Solution 3: text extraction + text classification

**Source**: [The code](https://github.com/richliao/textClassifier) is based on publication [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf). The code also has [a blog](https://richliao.github.io/supervised/classification/2016/12/26/textclassifier-HATN/) for detailed explanations.

**Data**:

**Method**:

Part 1: GRU-based sequence encoder

The GRU ([Gated Recurrent Unit](https://keras.io/api/layers/recurrent_layers/gru/)) uses a gating mechanism to track the state of sequences without using separate memory cells. There are two types of gates: the reset gate rt and the update gate zt. They together control how information is updated to the state.

Part 2: Hierarchical Attention

Hierarchical Attention includes Word Encoder, Word Attention, Sentence Encoder and Sentence Attention to achieve hierarchical modules.

Part 3: Softmax

Softmax dense layer for final Document Classification.


**Evalutation**: accuracy
