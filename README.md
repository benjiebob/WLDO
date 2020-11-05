# Who Left the Dogs Out?

Evaluation and demo code for our ECCV 2020 paper: *Who Left the Dogs Out? 3D Animal Reconstruction with Expectation Maximization in the Loop*.

- [Project page](https://sites.google.com/view/wldo/home)
- [Paper](https://arxiv.org/abs/2007.11110)

![](docs/banner.jpg)

Please note, this repository is in beta while I make bug fixes etc. Please let me know if you have problems or if anything is unclear -- it would also help me prioritise requests if you could let me know if you need your issue dealt with as part of your CVPR submission.

## Install

Clone the repository **with submodules**:

`git clone --recurse-submodules https://github.com/benjiebob/WLDO`

### Datasets

To use the StanfordExtra dataset, you will need to download the .json file [via the repository](https://github.com/benjiebob/StanfordExtra).

You may also wish to evaluate the [Animal Pose Dataset](https://sites.google.com/view/animal-pose/). If so, download 
all of the dog images into data/animal_pose/images. For example, an image path should look like: `data/animal_pose/images/2007_000063.jpg`.<sup>1</sup>

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

The results should read:

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


### Notes

<sup>1</sup> The eval script for animal_pose only runs on 94 of the 777 images in the dataset. This is because only these images
are provided with ground truth segmentations.

For segmentation decoding, install pycocotools
`python -m pip install "git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI"`
