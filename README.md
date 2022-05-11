# Scattering TinyImageNet-200
Some personal research code on analyzing deep learning models which works with scattering transform.

## Dataset
Dataset is Tiny ImageNet-200.Tiny ImageNet Challenge is the course project for Stanford CS231N. Tiny ImageNet has 200 classes and each class has 500 training images, 50 validation images, and 50 test images. The images sizes are 64 x 64 pixels. Test set doesn't contain labels therefore no results for them in this repo.

## Model
Scattering CNN model based on ResNet backbone and this backbone takes scattering transform output as input.\
Firstly a batch normalization applied to scattering output "nn.BatchNorm2d(in_channels, eps=1e-5, affine=False)"\
After normalization a 2D convolution comes for change channels value to 64 (this value determines according to multiple of 2, depends on architecture) from scattering output's channel value. Input is ready for usual ResNet implementation now.\
Additionally there is no downsampling for image resolutions because 2D scattering transform already reduces images size. Tiny Imagenet-200 images has 64x64 and after scattering images sizes decreased to 16x16.\

## Scattering
Refer kymatio website and github repository for get informations about scattering transforms and their implementation to python.
Web: https://www.kymat.io/
Github Repo: https://github.com/kymatio/kymatio

## Citation
> Andreux M., Angles T., Exarchakis G., Leonarduzzi R., Rochette G., Thiry L., Zarka J., Mallat S., Andén J., Belilovsky E., Bruna J., Lostanlen V., Hirn M. J., Oyallon E., Zhang S., Cella C., Eickenberg M. (2019). Kymatio: Scattering Transforms in Python. arXiv preprint arXiv:1812.11214.
