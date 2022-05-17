# Brain-arterial-segmentation
This repo utilizes U-Net to segment brain arteries from T1 weighted images. The UNet architecture was introduced for BioMedical Image segmentation by Olag Ronneberger et al. The introduced architecture had two main parts that were encoder and decoder. The encoder is all about the covenant layers followed by pooling operation. It is used to extract the factors in the image. The second part decoder uses transposed convolution to permit localization. It is again an F.C connected layers network. The original paper: [CovNet for Biomedical image segmentation](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28).
Here's an example of the U-net architecture (example for 32x32 pixels in the lowest resolution) given in the publication above.![](https://github.com/arrafi-musabbir/Brain-Arterial-Segmentation/blob/main/U-net.png)

## Ground truth generation
Subjects(1-20) data was collected from [ForrestGump dataset](https://openneuro.org/datasets/ds000113/versions/1.3.0). Arteries in T1 space can not be segmented with threshold value but it can be done in TOF space. So, the goal here became to register T1 in TOF space. We couldn’t register T1 in TOF directly as T1 and TOF have very different fields of view for which we register T2 in TOF as an intermediary image. Then registering T1 in T2 with T2 in TOF we were successfully able to register T1 in TOF. We also generated ground truth arterial segmentation by setting a threshold in TOF. [NeuroDebian](https://neuro.debian.net/) was used to preprocess the subjects before training on U-Net. 
### Axial view of T1 and TOF images
![T1 & TOF axial view](https://github.com/arrafi-musabbir/Brain-Arterial-Segmentation/blob/main/T1_TOF_axial.png)
### Skull stripped T2 images using [BET (Brain Extraction Tool)](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET) and aligning T2 arteries in TOF space and finally aligning T1 arteries in TOF space.
![T2 in TOF space](https://github.com/arrafi-musabbir/Brain-Arterial-Segmentation/blob/main/T2%20in%20TOF.png)
### T1 in TOF space
![T1 in TOF](https://github.com/arrafi-musabbir/Brain-Arterial-Segmentation/blob/main/T1%20in%20TOF.png)
### TOF arteries (grond truth)
![TOF arteries](https://github.com/arrafi-musabbir/Brain-Arterial-Segmentation/blob/main/TOF%20arteries.png).

The preprocessed dataset containing TOF_arteries and T1_in_TOF space for each subjects can be found from [this link](https://drive.google.com/drive/folders/16RUjW3yFcgrb6JM4kERk6r6Mbq5q4l9K?usp=sharing)

## Training U-net
U-net model was trained using sujects (1-15) TOF arteries. 
```python
BACKBONE = 'resnet34'
# preprocess_input = sm.get_preprocessing(BACKBONE)
model = sm.Unet(BACKBONE, input_shape=(None,None,3), encoder_weights='imagenet')
model.compile(
    'Adam',
    loss=sm.losses.bce_jaccard_loss,
    metrics=[sm.metrics.iou_score],
)

groundseg = nib.load(sub+'/tof_arteries.nii.gz').get_fdata()
t1 = nib.load(sub+'/t1_in_tof.nii.gz').get_fdata()
print('Training {}'.format(sub))
x_train, y_train, x_val, y_val = generatingInputs(t1, groundseg, sub)
model.fit(
    x=x_train,
    y=y_train,
    batch_size=16,
    epochs=100,
    validation_data=(x_val, y_val)
)

```
Trained model can be found from [this link](https://drive.google.com/file/d/1-0WJcb9Yzay1b6v_nDM3mK0qy1M2zoG-/view?usp=sharing).

## Using trained model to test on Sujects (15-20)
### Loading pretrained model
```python
model = tf.keras.models.load_model('Final model.h5', custom_objects={'binary_crossentropy_plus_jaccard_loss': sm.losses.bce_jaccard_loss, 'iou_score': sm.metrics.iou_score})
```
### Prediction
```python
groundseg = nib.load('sub-16/tof_arteries.nii.gz').get_fdata()
t1   = nib.load('sub-16/t1_in_tof.nii.gz').get_fdata()
t1, groundseg = reshaping(t1, groundseg)
preds = model.predict(t1)
```
![Testing on Subject 16](https://github.com/arrafi-musabbir/Brain-Arterial-Segmentation/blob/main/s16%20prediction.png)
