# Thermal Chameleon Net: Task-Adaptive Tone-mapping for Thermal-Infrared images

Official Repository for "Thermal Chameleon Net: Task-Adaptive Tone-mapping for Thermal-Infrared images", Underreview

![thermal_cameleon](https://github.com/ThermalCameleon/ThermalCameleonNet/assets/150974352/e82e98e8-63f7-477b-bd0d-35efc02c6481)

Above is a picture of thermal cameleon that we've made using Dall-E.

TLDR: We propose a new task-adaptive learnable tone-mapping network for thermal infrared images from 14-bit (RAW) thermal infrared images. 

### Why bother using 14-bit thermal infrared images??

Thermal images we usually see from datasets look something similar to the image below. 


- Some common strategies to counteract this it to use different 14-bit to 8-bit rescaling methods like below but that's still using quantized 8-bit images. 
![main_figure_overview](https://github.com/ThermalCameleon/ThermalCameleonNet/assets/150974352/60a02d15-0f65-44a6-a236-71af77b23a44)

In our work, we decided to employ 14-bit thermal images directly as inputs to neural networks to get the best out of the original data!
![detection_results_snu](https://github.com/ThermalCameleon/ThermalCameleonNet/assets/150974352/5ea2f646-87f4-4c71-8a54-f9aaa306b324)

The figure above shows the perceived image from the feature map prior to the object detection input. Essentially it is what image the object detection networks like to see in thermal image-based object detection. 

### Overview of the Thermal Cameleon Network

<div align="center">
  
![overview_proposed](https://github.com/ThermalCameleon/ThermalCameleonNet/assets/150974352/62069c32-1366-4bbb-a7ee-2fd1347b995e)

</div>

Just like the name states, our work is aimed at creating object detection adaptive network from 14-bit thermal images. 

Our method is divided into two stages:

- Multichannel thermal embedding: Essentially a tool to represent each absolute temperature value (in Celsius) to a set feature vectors. 
- Adaptive channel compression network: Employing lots of multichannel embeddings always don't work and it even incurs high computational cost. More importantly, we can't use transfer learning this way as they are optimized for 3 channel inputs. This essentially enables all those operations by compressing only valid features for object detection in three channel representations. 


# Results
### Quantitative Results on object detection

<div align="center">
  
![table](https://github.com/ThermalCameleon/ThermalCameleonNet/assets/150974352/4c543b54-7e92-4b11-b449-62428c20c6f3)

</div>


### Qualitative on object detection across multiple datasets

<div align="center">
  
![comparisons](https://github.com/ThermalCameleon/ThermalCameleonNet/assets/150974352/966a72f6-f65d-421b-9139-2cf6d19d55d7)

</div>

Cyan refers to 'car' class and Yello refers to 'person' class. 

## Code will become available after the end of blind review period. 
