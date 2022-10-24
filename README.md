# MEW-UNet
This is the official code repository for "Multi-axis Representation Learning in Frequency Domain for Medical Image Segmentation"

**0. Main Environments**
- python 3.8
- pytorch 1.8.0
- torchvision 0.8.0

**1. Prepare the dataset and our weights.**

- Synapse dataset can be found at [the repo of TransUnet](https://github.com/Beckschen/TransUNet). 

- Our weights of the Synapse dataset can be download at [here](https://pan.baidu.com/s/1ERqX_qBhag4xlZ5Wh6RnCw), with a password of *ex8x*

- After download the weights, you could put weights into this file('./our_weights/')

**2. Test data folder format**

- data
  - Synapse
    - test_vol_h5
      - case0001.npy.h5
      - case0002.npy.h5
      - case0003.npy.h5
      - case0004.npy.h5
      - case0008.npy.h5
      - case0022.npy.h5
      - case0025.npy.h5
      - case0029.npy.h5
      - case0032.npy.h5
      - case0035.npy.h5
      - case0036.npy.h5
      - case0038.npy.h5

**3. Test our model.**

```
cd MEW-UNet
```
```
python test.py
```

After testing about 20 mins, you can obtain the results in './test_log/test_log/'



