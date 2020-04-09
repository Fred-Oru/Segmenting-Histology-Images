# Glomeruli Segmentation in Histology images

This repository is an Artificial Intelligence project aiming to segment [kidney glomeruli](https://en.wikipedia.org/wiki/Glomerulus_(kidney)) in histology images. It makes use of the [Mask R-CNN implementation by Matterport](https://github.com/matterport/Mask_RCNN).

Performance as of 9th of April 2020 : AP ~ 0.75, Mask IoU ~ 0.88

![Example of segmented image](DataSamples/segmented_image.png)

# Table of Contents
<!-- TOC depthFrom:2 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Business Problem](#business-problem)
- [Data specifications](#data-specifications)
- [Available Data](#available-data)
- [Project Definition](#project-definition)
- [Performance Metrics](#performance-metrics)
- [Results Log](#results-log)

<!-- /TOC -->

## Business Problem

Des chercheurs en biologie étudient les effets d'une pathologie sur les [glomérules](https://fr.wikipedia.org/wiki/Glom%C3%A9rule_r%C3%A9nal) du rein. Pour mesurer ces effets, ils étudient des coupes histologiques de reins sur lesquels la pathologie est révélée par un marquage de couleur qui ressort généralement en marron foncé sur un fond marron clair. Pour quantifier le niveau de pathologie du rein, les chercheurs doivent estimer le pourcentage de la surface des glomérules qui est marquée en marron foncé.

A l'aide d'un logiciel de gestion d'images (Distribution [Fiji](http://fiji.sc/) du logiciel open source [Imaji](https://en.wikipedia.org/wiki/ImageJ)), les chercheurs :
* délimitent manuellement les contours des glomérules (précisément la membrane basale de la capsule de Bowman),
* font un seuillage pour ne garder que la surface marquée en marron foncé,
* mesurent à l'aide du logiciel la surface marquée et la surface totale du glomérule pour en déduire le pourcentage marqué.

L'opération la plus fastidieuse et chronophage est celle qui consiste à délimiter manuellement les glomérules. L'objectif du projet est d'entraîner une intelligence artificielle à segmenter automatiquement les glomérules dans une image histologique de rein.

## Data specifications

Les coupes histologiques sont livrées aux chercheurs sous la forme d'un fichier numérique de très haute résolution, dans un format propriétaire qui ne peut être visualisé qu'avec un logiciel spécifique. Pour des raisons de taille et de format, il n'est pas possible de procéder aux mesures directement sur cette image. En revanche, il est possible de zoomer sur des zones de l'image et les exporter au format jpg.
Par ailleurs, les chercheurs ne s'intéressent qu'à certaines zones de l'image, là où la coupe est nette et montre une section significative des glomérules.

Pour toutes ces raisons, les chercheurs commencent par extraire de l'image haute résolution des zooms sur des zones d'intérêts.

Exemple (cf. dossier Data/sample) :
* les fichiers R22 VEGF 2_15.0xC1.jpg à R22 VEGF 2_15.0xC5.jpg sont des images extraites d'une même coupe histologique du rein n° R22
* VEGF est un des types de marqueurs utilisés. D'autres coupes sont effectuées avec des reins marqués par PAS ou HES.
* 15 est le grossissement du zoom par rapport à l'image haute définition d'origine. Les chercheurs peuvent utiliser des grossissement entre 5 et 30.
* le fichier a typiquement une résolution de 1920x1018 pixels et pèse environ 2,5 Mo. Il peut cependant y avoir des résolutions légèrement différentes.


Dans ImageJ, les chercheurs ouvrent chaque image zoom, et délimitent les cellules à l'aide d'une souris ou d'une tablette graphique. Chaque délimitation est enregistrée par Fiji sous la forme d'un fichier .roi (acronyme de "Region of interest"). Il y a donc généralement plusieurs fichiers ROI pour chaque image zoom, et ils sont tous compressés dans un .zip
* Par exemple : RoiSetR22C1.zip contient sept ROI correspondant à des cellules dans l'image R22 VEGF 2_15.0xC1.jpg
* Chaque ROI contient (entre autres) les coordonnées (x,y) de chaque pixel du contour délimité par le chercheur.

Une fois les ROI générés, le logiciel est capable de calculer la surface du glomérule et la surface marquée à l'intérieur du glomérule.

## Available Data

Pour des raisons de confidentialité et de taille de stockage, le repository github ne présente qu'un échantillon de données (dossier Data/sample). L'ensemble des données est accessible aux contributeurs du projet sur un [drive google partagé](https://drive.google.com/open?id=1rmJG8g-bZpiiZyb6SJd3uqtqJOa-EQ9X).

* Dossier Data/glomerulus/train (500 Mo):
  * 200 images avec les fichiers .roi associés
  * résolution 1920x1018  en général, quelquefois 1831x1058


* Dossier Data/glomerulus/test (175 Mo):
  * 80 images

A ce jour, le train dataset ne contient exclusivement que des colorations VEGF au grossissement x15. Il sera possible de récupérer des données plus diversifiées post-confinement.

## Project Definition

L'objectif de ce projet d'intelligence artificielle est :
* de prendre en entrée une image zoom de taille de de grossissement quelconque
* de produire en sortie
  * un fichier zip de ROI contenant les délimitations de chaque cellule dans l'image
  * une image avec les délimitations superposées

## Performance Metrics

Performance for instance segmentation models are usually measured with [Mean Average Precision](https://medium.com/@yanfengliux/the-confusing-metrics-of-ap-and-map-for-object-detection-3113ba0386ef). However this metrics is very technical and does not reflect the business perspective.

User interviews about their expectations revealed that :
* They expect the program to detect all glomeruli
* They can accept approx. 10% error on the contour (as they themselves don't draw perfect contours)
* They don't mind false postive too much (it's easy to discard them)

As a consequence, the performance metrics are defined as :
* % of undetected glomeruli
* % of false positive
* Mean IoU between generated masks and ground truth mask for True Positive

## Results Log

Version 0 :
* Training dataset is composed of 200 VEGF x15 images.
* Test dataset contains very different markers and scales

V0.2 - 2020/04/09

|                | Train       | Valid       | Test |
|----------------|-------------|-------------|------|
|#images         | 161         | 39          |      |
|#glomeruli      | 1161        | 277         |      |
|#undetected     |2 (0.17%)    | 2 (0.72%)   |      |
|#false positive |118 (10.16%) | 19 (6.86%)  |      |
|#mean IoU on TP |0.8761       | 0.8852      |      |
|#mean AP        |0.7639       | 0.7571      |      |


V0.1 - 2020/03/23

|                | Train       | Valid       | Test |
|----------------|-------------|-------------|------|
|#images         | 161         | 39          |      |
|#glomeruli      | 1161        | 277         |      |
|#undetected     |1 (0.09%)    | 1 (0.36%)   |      |
|#false positive |162 (13.95%) | 35 (12.64%) |      |
|#mean IoU on TP |0.6906       | 0.8349      |      |
|#mean AP        |0.7194       | 0.7070      |      |
