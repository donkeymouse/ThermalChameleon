## Thermal Chameleon Net: Task-Adaptive Tone-mapping for Thermal-Infrared images

Official Repository for "Thermal Chameleon Net: Task-Adaptive Tone-mapping for Thermal-Infrared images", Underreview

![tc](https://github.com/donkeymouse/ThermalChameleon/assets/58677731/fc46250b-e4df-41c3-8f1b-d69e8a1269f7)

Above is a picture of thermal cameleon that we've made using Dall-E.

TLDR: We propose a new task-adaptive learnable tone-mapping network for thermal infrared images from 14-bit (RAW) thermal infrared images. 

### Abstract

Thermal Infrared (TIR) imaging offers robust perception for navigating challenging outdoor environments. However, it encounters significant challenges, notably in texture and contrast, stemming from its typical 14/16-bit format. Traditional rescaling methods, aimed primarily at geometric tasks like depth estimation, lack flexibility for semantic downstream tasks and require extensive environmental knowledge for image standardization. To bridge this gap, we present the Thermal Chameleon Network (TCNet), an innovative approach to task-adaptive tone-mapping for RAW 14-bit TIR images. TCNet introduces multichannel thermal embedding, eliminating the need for heuristic image rescaling, and incorporates adaptive channel compression to streamline the embedding into a 3-channel output. This advancement facilitates task-specific tone-mapping and ensures TCNet's modularity with diverse network architectures and applications. Demonstrating minimal computational increase and superior adaptability to limited training data, TCNet excels in object detection, setting new horizons for improved semantic task performance in TIR imaging.

Too long to read? Here's a TL;DR

**Don't spend time on tone-mapping thermal images that would work well for all tasks, instead let the network do it for you, optimized for each task!**

### Overview of the Thermal Cameleon Network

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

<details>
  <summary> Object detection </summary>
  
  <details>
  <summary> RetinaNet </summary>
  
    * item1
    * itme2

  </details>

</details>


## Usage

### Will be announced after review period
