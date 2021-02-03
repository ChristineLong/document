# Scanned document classification: 

## Method 1: image classification with inter and intra domain 
https://github.com/arpan65/Scanned-document-classification-using-deep-learning

**Data**: [RVL-CDIP (Ryerson Vision Lab Complex Document Information Processing) data set](https://www.cs.cmu.edu/~aharley/rvl-cdip/) which consists of 400,000 grayscale images in 16 classes, with 25,000 images per class. There are 320,000 training images, 40,000 validation images, and 40,000 test images. The images are sized so their largest dimension does not exceed 1000 pixels. The size of this data set is more than 200 GB. Address is rvl-cdip.tar

**Method**:
1. Import image data with ImageDataGenerator (from tf.keras.preprocessing.image): N_generated_img = number of epochs * batch size * steps_per_epoch with the standard given in the method. Data augmentation is optional here due to little change to result.
2. Uses inter and intra domain transfer learning where an image is divided into four parts header, footer, left body and right body. 
  + Common: A pretrained VGG16 model is first used to train over the whole images(inter domain) then this model is used to train the part of images(Intra domain).
  + Here: Instead of intra domain transfer learning using VGG16, we trained two parallel models VGG16 and InceptionResNetV2 and used a stack of these as our final model. Our assumption was that because of the different architectures of these two models they will learn the different aspect of images and stacking them will result in good generalization.
3. Tuning hyperparameters: For any CNN the hyper parameters are: learning rate, pooling size, network size, batch size, choice of optimizer, regularization, input size etc.
  + learning rate: here use  ‘Cyclic Learning Rate’, which aims to train neural network such a way that the learning rate changes in a cyclic way for each training batch. It varies the learning rate within a threshold. The periodic higher learning rate helps to overcome if it stuck in the saddle point or local minima.

**Evaluation**: Accuracy, micro average F1 score, confusion metric for all classes in heatmap (See repo). Training accuracy is 97% and test accuracy is 91.45%.

why not OCR to extract text and apply NLP techniques: _Low quality scans resulted in a poor quality of text extraction_. In the practical business scenarios also we do not have control over the quality of scans, so models rely on OCR may suffer from poor generalization even after proper preprocessing.





## Method 2: multi-modal Transformer model with document text + layout + image information 
placeholder

## Method 3: image object detection + empirical rules
placeholder

## Method 4: image classification with localization
placeholder

