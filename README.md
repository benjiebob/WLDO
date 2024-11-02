# Who Left the Dogs Out?

Evaluation and demo code for our ECCV 2020 paper: *Who Left the Dogs Out? 3D Animal Reconstruction with Expectation Maximization in the Loop*.

- [Project page](https://sites.google.com/view/wldo/home)
- [Paper](https://arxiv.org/abs/2007.11110)

![](docs/banner.jpg)

## Install

Clone the repository **with submodules**:

`git clone --recurse-submodules https://github.com/benjiebob/WLDO`


For segmentation decoding, install pycocotools
`python -m pip install "git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI"`


### Datasets

Follow the instructions [on the repository](https://github.com/benjiebob/StanfordExtra) to download StanfordExtra annotations and images.

**Please ensure you have StanfordExtra_v12 installed, which we released 1 Feb 2021.**

You may also wish to evaluate the [Animal Pose Dataset](https://sites.google.com/view/animal-pose/). If so, download 
all of the dog images into data/animal_pose/images. For example, an image path should look like: `data/animal_pose/images/2007_000063.jpg`. We have reformatted the annotation file and enclose it in this repository `data/animal_pose/animal_pose_data.json`.

## Splits

The train/validation/test splits used for our ECCV 2020 submission are contained in the `data/StanfordExtra_v12` repository and under the `data/animal_pose` folder.

### Pretrained model

Please download our [pretrained model](https://drive.google.com/file/d/1khc-wttwBZ-I2ub1OgkhdB9aFDo0OMn4/view?usp=sharing) and place underneath `data/pretrained/3501_00034_betas_v4.pth`.

# Quickstart

## Demo

To run the model on a series of images, place the images in a directory, and call the script demo.py.
To see an example of this working, run demo.py and it will use the images in `example_imgs`:

```
cd wldo_regressor
python demo.py
```

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

<table>
  <thead>
  <tr>
    <th>Dataset</th>
    <th>IOU</th>
    <th colspan="5">PCK @ 0.15</th>
  </tr>
  <tr>
    <th></th>
    <th></th>
    <th>Avg</th>
    <th>Legs</th>
    <th>Tail</th>
    <th>Ears</th>
    <th>Face</th>
  </tr>
  </thead>
  <tbody>
  <tr>
    <td>StanfordExtra</td>
    <td>74.2</td>
    <td>78.8</td>	
    <td>76.4</td>	
    <td>63.9</td>	
    <td>78.1</td>	
    <td>92.1</td>
  </tr>
  <tr>
    <td>Animal Pose</td>
    <td>67.5</td>	
    <td>67.6</td>	
    <td>60.4</td>	
    <td>62.7</td>	
    <td>86.0</td>	
    <td>86.7</td>
  </tr>
  </tbody>
</table>

Note that we have recently updated the tables in the arxiv version of our paper to account for some fixed dataset annotations and to use an improved version of the PCK metric. More details can be found in the paper.


## Related Work
This repository owes a great deal to the following works and authors:
- [SMALify](https://github.com/benjiebob/SMALify/); Biggs et al. provided an energy minimization framework for fitting to animal video/images. A version of this was used as a baseline in this paper.
- [SMAL](http://smal.is.tue.mpg.de/); Zuffi et al. designed the SMAL deformable quadruped template model and have provided me with wonderful advice/guidance throughout my PhD journey.
- [SMALST](https://github.com/silviazuffi/smalst); Zuffi et al. provided PyTorch implementations of the SMAL skinning functions which have been used here.

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
(c) Benjamin Biggs, Oliver Boyne, Andrew Fitzgibbon and Roberto Cipolla. Department of Engineering, University of Cambridge 2020

As of 02-NOV-2024, this dataset is now MIT licensed. Enjoy!

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
