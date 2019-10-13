# Malaria Detection
Create a transfer learning CNN model based on AlexNet to detect malaria through cell images

**Techniques used:**    

1.**Color Constancy** 

2.**Convolutional Neural Network**  

3.**AlexNet(Transfter learning)**

**Requirement:** 

- Matlab 2018b

## Data

Download the original data from U.S. National Library of Medicine, link below:
https://ceb.nlm.nih.gov/repositories/malaria-datasets/

## Project Methodology

![flowchart](/image/flowchart.png)

## Orginal cell images (parasitized & uninfected)

![originparasitized](/image/originparasitized.png)

![originuninfected](/image/originuninfected.png)

## Dataset Distribution

![datadistribution](/image/datadistribution.png)

![finaldatadistribution](/image/finaldatadistribution.png)

## Image Processing (Color Constancy)

![processedcell1](/image/processedcell1.png)

![processedcell2](/image/processedcell2.png)

## Result
**Confusion Matrix (w/without transfer learning):**
![confusionnotrans](/image/confusionnotrans.png)

![confusiontrans](/image/confusiontrans.png)
**ROC (w/without transfer learning):**
![ROCnotrans](/image/ROCnotrans.png)

![ROCtrans](/image/ROCtrans.png)

**Training Process (w/without transfer learning):**
![trainingnotrans](/image/trainingnotrans.png) 

![trainingtrans](/image/trainingtrans.png) 

