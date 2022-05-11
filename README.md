# Scattering TinyImageNet-200
Some personal research code on analyzing deep learning models which works with scattering transform.

## Dataset
Dataset is Tiny ImageNet-200. Tiny ImageNet Challenge is the course project for Stanford CS231N. Tiny ImageNet has 200 classes and each class has 500 training images, 50 validation images, and 50 test images. The images sizes are 64 x 64 pixels. Test set doesn't contain labels therefore no results for them in this repo.

## Model
Scattering CNN model based on ResNet backbone and this backbone takes scattering transform output as input.\
Firstly a batch normalization applied to scattering output "nn.BatchNorm2d(in_channels, eps=1e-5, affine=False)"\
After normalization a 2D convolution comes for change channels value to 64 (this value determines according to multiple of 2, depends on architecture) from scattering output's channel value. Input is ready for usual ResNet implementation now.\

Additionally there is no downsampling for image resolutions because 2D scattering transform already reduces images size. Tiny Imagenet-200 images has 64x64 and after scattering images sizes decreased to 16x16.

## Scattering
Refer kymatio website and github repository for get informations about scattering transforms and their implementation to python.\
Web: https://www.kymat.io/ \
Github Repo: https://github.com/kymatio/kymatio
 
## Citation
> Andreux M., Angles T., Exarchakis G., Leonarduzzi R., Rochette G., Thiry L., Zarka J., Mallat S., Andén J., Belilovsky E., Bruna J., Lostanlen V., Hirn M. J., Oyallon E., Zhang S., Cella C., Eickenberg M. (2019). Kymatio: Scattering Transforms in Python. arXiv preprint arXiv:1812.11214.

## Results
Model trained from scratch and 51.54% validation accuracy achieved in 15 epochs.
![15epoch_acc51_54](https://user-images.githubusercontent.com/86148100/167864740-a1e80675-4604-449b-b03e-2fb1ed96a72e.png)

Loss Graph:
![15epoch_loss](https://user-images.githubusercontent.com/86148100/167864788-f758bf0f-5279-4491-967f-6c47817f86c1.png)

## Future Work
Model contains ResNet backbone but the old works shows that usual ResNet models can not achieve 50% validation accuracy from scratch in any epoch.\
Check out this sites to see the results for different models with this dataset.\
ResNet-50: https://github.com/aqua1907/tiny_imagenet \
ResNet-18: https://github.com/tjmoon0104/Tiny-ImageNet-Classifier \
AlexNet: https://github.com/DennisHanyuanXu/Tiny-ImageNet \

According to the Standford University report, Inception-ResNet model can achieve 56.9% validation accuracy.\
http://cs231n.stanford.edu/reports/2017/pdfs/930.pdf

This article used DenseNet and achieved 62.7% accuracy.\
https://arxiv.org/ftp/arxiv/papers/1904/1904.10429.pdf

Scattering transform should be tried with this backbones.
