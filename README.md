# Who Left the Dogs Out?

Evaluation and demo code for our ECCV 2020 paper: *Who Left the Dogs Out? 3D Animal Reconstruction with Expectation Maximization in the Loop*.

- [Project page](https://sites.google.com/view/wldo/home)
- [Paper](https://arxiv.org/abs/2007.11110)

![](docs/banner.jpg)

## DISCLAIMER

Please note, this repository is in beta while I make bug fixes etc. Please let me know if you have problems or if anything is unclear -- it would also help me prioritise requests if you could let me know if you need your issue dealt with as part of your CVPR submission.

## Install

Clone the repository **with submodules**:

`git clone --recurse-submodules https://github.com/benjiebob/WLDO`

### Datasets

To use the StanfordExtra dataset, you will need to download the .json file [via the repository](https://github.com/benjiebob/StanfordExtra).

You may also wish to evaluate the [Animal Pose Dataset](https://sites.google.com/view/animal-pose/). If so, download 
all of the dog images into data/animal_pose/images. For example, an image path should look like: `data/animal_pose/images/2007_000063.jpg`.<sup>1</sup>

## Splits
Our train/test splits are contained in the `data/splits` repository. Since ECCV 2020 we have sourced annotations for additional images, contained in the StanfordExtra repository which you can use for training/test. However, the npy files in `data/splits` are the same as we used in the paper.

### Pretrained model

Please download our [pretrained model](https://drive.google.com/file/d/1n-aZk5x9cvrwB8QR6SeeZnI-rHrro_YP/view?usp=sharing) and place underneath `data/pretrained/3501_00034_betas_v2.pth`.

# Quickstart

## Eval

To evaluate the performance of the model on the StanfordExtra dataset, run eval.py:

```
cd wldo_regressor
python eval.py --dataset stanford
```

You can also run on the `animal_pose` dataset

```
python eval.py --dataset animal_pose
```

## Results

The results of this model are slightly improved from the paper, due to some minor fixes to the annotations in the StanfordExtra dataset (generally resourcing annotations for train/test images we noticed were of very poor quality). This process has generally improved performance, although our result for Animal Pose PCK has dropped slightly. However, note that we do not use Animal Pose for training and it contains only ~94 usable test images (e.g. has segmentation, 1 dog per image etc.) compared to the ~1.7k we used in the original StanfordExtra test split.

<table>
  <thead>
  <tr>
    <th>Dataset</th>
    <th colspan="5">PCK</th>
    <th>IOU</th>
  </tr>
  <tr>
    <th></th>
    <th>Avg</th>
    <th>Legs</th>
    <th>Tail</th>
    <th>Ears</th>
    <th>Face</th>
    <th></th>
  </tr>
  </thead>
  <tbody>
  <tr>
    <td>StanfordExtra</td>
    <td>82.8</td>
    <td>79.1</td>
    <td>74.6</td>
    <td>84.5</td>
    <td>95.6</td>
    <td>74.8</td>
  </tr>
  <tr>
    <td>Animal Pose</td>
    <td>67.6</td>
    <td>61.0</td>
    <td>62.7</td>
    <td>83.3</td>
    <td>91.1</td>
    <td>66.6</td>
  </tr>
  </tbody>
</table>

## Demo

To run the model on a series of images, place the images in a directory, and call the script demo.py.
To see an example of this working, run demo.py and it will use the images in `example_imgs`:

```
cd wldo_regressor
python demo.py
```

## Related Work
This repository owes a great deal to the following works and authors:
- [SMALify](https://github.com/benjiebob/SMALify/); Biggs et al. provided an energy minimization framework for fitting to animal video/images. A version of this was used as a baseline in this paper.
- [SMAL](http://smal.is.tue.mpg.de/); Zuffi et al. designed the SMAL deformable quadruped template model and have been wonderful for providing advice throughout my animal reconstruction PhD journey.
- [SMALST](https://github.com/silviazuffi/smalst); Zuffi et al. provided a PyTorch implementations of the SMAL skinning functions which have been used here.
- [SMPLify](http://smplify.is.tue.mpg.de/); Bogo et al. provided the basis for our original ChumPY implementation and inspired the name of this repo.

## Acknowledgements

If you make use of this code, please cite the following paper:

```
@inproceedings{biggs2020wldo,
  title={{W}ho left the dogs out?: {3D} animal reconstruction with expectation maximization in the loop},
  author={Biggs, Benjamin and Boyne, Oliver and Charles, James and Fitzgibbon, Andrew and Cipolla, Roberto},
  booktitle={ECCV},
  year={2020}
}
```

## Contribute
Please create a pull request or submit an issue if you would like to contribute.

## Licensing
(c) Benjamin Biggs, Oliver Boyne, James Charles, Andrew Fitzgibbon and Roberto Cipolla. Department of Engineering, University of Cambridge 2020

By downloading this code, you agree to the [Creative Commons Attribution 3.0 International license](https://creativecommons.org/licenses/by/3.0/). This license allows users to use, share and adapt the code, so long as credit is given to the authors (e.g. by citation).

THIS SOFTWARE AND ANNOTATIONS ARE PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

### Notes

<sup>1</sup> The eval script for animal_pose only runs on 94 of the 777 images in the dataset. This is because only these images
are provided with ground truth segmentations.

For segmentation decoding, install pycocotools
`python -m pip install "git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI"`
