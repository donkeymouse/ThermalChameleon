## Thermal Chameleon: Task-Adaptive Tone-mapping for Radiometric Thermal-Infrared images

Official Repository for "Thermal Chameleon Net: Task-Adaptive Tone-mapping for Thermal-Infrared images", Robotics and Automation Letters (RA-L). 

<div align="left">  
  <a href="https://scholar.google.com/citations?user=u6VDnlgAAAAJ&hl=ko&oi=ao">Dong-Guw Lee</a>,  
  <a href="https://scholar.google.com/citations?hl=ko&user=vW2JtFAAAAAJ">Jeongyun Kim</a>,
  <a href="https://scholar.google.com/citations?user=W5MOKWIAAAAJ&hl=ko&oi=ao">Younggun Cho</a>,  
  <a href="https://ayoungk.github.io/">Ayoung Kim</a>
</div>


![tc](https://github.com/donkeymouse/ThermalChameleon/assets/58677731/fc46250b-e4df-41c3-8f1b-d69e8a1269f7)

Above is a picture of thermal chameleon that we've made using Dall-E.

TLDR: We propose a new task-adaptive learnable tone-mapping network for thermal infrared images from 14-bit (RAW) thermal infrared images. 

### Abstract

Thermal Infrared (TIR) imaging provides robust perception for navigating in challenging outdoor environments but faces issues with poor texture and low image contrast due to its 14/16-bit format. Conventional methods utilize various tone-mapping methods to enhance contrast and photometric consistency of TIR images, however, the choice of tone-mapping is largely dependent on knowing the task and temperature dependent priors to work well. In this paper, we present Thermal Chameleon Network (TCNet), a task-adaptive tone- mapping approach for RAW 14-bit TIR images. Given the same image, TCNet tone-maps different representations of TIR images tailored for each specific task, eliminating the heuristic image rescaling preprocessing and reliance on the extensive prior knowledge of the scene temperature or task-specific characteris- tics. TCNet exhibits improved generalization performance across object detection and monocular depth estimation, with minimal computational overhead and modular integration to existing architectures for various tasks. 

Too long to read? Here's a TL;DR

**Don't spend time on tone-mapping thermal images that would work well for all tasks, instead let the network do it for you, optimized for each task!**

### Overview of Thermal Chameleon

<div align="center">
  
![method_overview](https://github.com/donkeymouse/ThermalChameleon/assets/58677731/34bf3528-32b4-4282-9039-fbffdbc56274)


</div>

Just like the name states, our work is aimed at creating object detection adaptive network from 14-bit thermal images. 

Our method is divided into two stages:

- Multichannel thermal embedding: Essentially a tool to represent each absolute temperature value (in Celsius) to a set feature vectors. 
- Adaptive channel compression network: Employing lots of multichannel embeddings always don't work and it even incurs high computational cost. More importantly, we can't use transfer learning this way as they are optimized for 3 channel inputs. This essentially enables all those operations by compressing only valid features for object detection in three channel representations. 

In essence, what really happens is that we assign task-adaptive weights to each thermal embedding, optimized and controlled by the loss functions of the downstream task.

![task_adaptive_tonemapv2](https://github.com/donkeymouse/ThermalChameleon/assets/58677731/e333fe24-d0ad-410e-ad48-0d2cbea84663)



# Results
### Quantitative Results on object detection

<details>
  <summary>FLIR-ADAS Dataset</summary>
  
<div align="center">
  
![obj_det_flir](https://github.com/donkeymouse/ThermalChameleon/assets/58677731/d1b1e5ae-2d5d-42bf-9bfd-3ce5881d0ecb)


</div>

</details>

<details>
  <summary>Zero-shot detection on various public dataset</summary>
  
<div align="center">
  
![obj_det_unseen](https://github.com/donkeymouse/ThermalChameleon/assets/58677731/423ac135-4c01-45c4-8ded-62aaf65f2dae)

</div>

</details>


### Qualitative on depth estimation


<details>
  <summary> VIVID Dataset </summary>
  
<div align="center">
  
![VIVID](https://github.com/donkeymouse/ThermalChameleon/assets/58677731/f2012c3c-27aa-45d1-88dd-ea302230507b)


</div>

</details>

<details>
  <summary>Zero-shot detection on FLIR-ADAS/STheReO </summary>
  
<div align="center">
  
![unseen_depth](https://github.com/donkeymouse/ThermalChameleon/assets/58677731/66b98431-e098-4ae9-ba02-136ec8d18192)


</div>

</details>


## Training Details

### Basic setting

* All settings use ResNet50 as backbones unless specified
* All models were trained using Nvidia RTX 4090 / RTX-A6000 for YOLOX
* All models were trained for 500 epochs with weights being saved for each epoch. We took the best epoch based on the validation set. 

### Object detection 

<details>
  <summary> RetinaNet </summary>
  
  * Warm up epoch: 10
  * Batch size: 16
  * Optimizer: AdamW
  * Base lr: $1.5 \times 10^{-4}$
  * Scheduler: Cosine annealing
  * Data augmentation: Random horizontal flip
  * Pretraining?: No (Trained from scratch)

</details>

<details>
  <summary> YOLOX </summary>
  
  * Warm up epoch: 5
  * Batch size: 32
  * Optimizer: SGD with momentum of 0.9
  * Weight decay: 0.05
  * Base lr: $1.5625 \times 10^{-4}$
  * Scheduler: Cosine annealing
  * Data augmentation: Random horizontal flip, Random mosaic, Random mixup
  * Pretraining?: No (Trained from scratch)
  Pretty much all settings are identical to original YOLO-X implementations.

</details>

<details>
  <summary> Sparse-RCNN </summary>

  Implemented on MMDetection
  
  * Warm up iterations: 1000 iterations
  * Batch size: 16
  * Optimizer: AdamW 
  * Weight decay: 0.0001
  * Base lr: $2.5 \times 10^{-4}$
  * Scheduler: Cosine annealing
  * Data augmentation: Random horizontal flip, Random mosaic, Random mixup
  * Pretraining?: Yes (ImageNet pretraining). For Thermal embedding, we averaged out the 3 channel weights and copied it to all channels for the first conv layer.

</details>


### Depth estimation 

<details>
  <summary> Monodepth-Thermal </summary>
  
  * Batch size: 4
  * Optimizer: Adam
  * Base lr: $1.5 \times 10^{-4}$
  * Scheduler: Cosine annealing
  * Data augmentation: Random horizontal flip/Random crop
  * Pretraining?: Yes (ImageNet Pretraining)

  
  Followed all protocols and most settings used in this repo: https://github.com/UkcheolShin/ThermalMonoDepth

</details>



## Usage

### Will be announced after review period


## Citation

Please consider citing the paper as:
```
@ARTICLE {dglee-2024-tcnet,
    AUTHOR = { Dong-Guw Lee and Jeongyun Kim and Younggun Cho and Ayoung Kim },
    TITLE = { Thermal Chameleon: Task-Adaptive Tone-mapping for Radiometric Thermal-Infrared images },
    JOURNAL = {IEEE Robotics and Automation Letters (RA-L) },
    YEAR = { 2024 },
}

```  

## Contact
If you have any *urgent* questions or issues that need to be resolved, please contact me by email. 
```
donkeymouse@snu.ac.kr
```
