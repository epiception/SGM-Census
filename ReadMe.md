SGM-Census
==========

Simple CPU implementation of Semi-Global Matching using Census Transform and Hamming Distance Matching Cost.

<img src="Examples/output_montage.png" style="width: 25%;"/>

---
## Instructions

    $git clone https://github.com/epiception/SGM-Census.git
    $cd ~/path/to/SGM-Census
    $make

**Usage**

    ./sgm <right image> <left image> <output image file> <disparity range>

**Examples**

    ./sgm Examples/teddy/right.png Examples/teddy/left.png output_disparity.png 64

---
#### Resources

* [Accurate and Efficient Stereo Processing by Semi-Global Matching and Mutual Information](http://www.dlr.de/rm/en/PortalData/3/Resources/papers/modeler/cvpr05hh.pdf)
* [Semi-Global Matching - Motivation, Developments and Applications](http://www.ifp.uni-stuttgart.de/publications/phowo11/180Hirschmueller.pdf)
* [Semi-Global Matching](http://lunokhod.org/?p=1356)
* [Middlebury Stereo Datasets](http://vision.middlebury.edu/stereo/data/)
* [Kitti Stereo Dataset](http://www.cvlibs.net/datasets/kitti/eval_stereo.php)

#### References
* Main Reference: https://github.com/reisub/SemiGlobal-Matching
* Terminal progress bars: https://github.com/luigipertoldi/progressbar

