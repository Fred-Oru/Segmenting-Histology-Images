# Segmenting-Histology-Images

This repository is an Artificial Intelligence project aiming at segmenting cells in histology images.

## Business Problem

Des chercheurs en biologie reçoivent des coupes histologiques d'animaux. Ils souhaitent calculer le nombre et la surface occupée par certains types de cellules sur ces coupes. 

A l'aide d'un logiciel de gestion d'images (Distribution [Fiji](http://fiji.sc/) du logiciel open source [Imaji](https://en.wikipedia.org/wiki/ImageJ), les chercheurs délimitent manuellement les cellules visibles pour en calculer la surface. L'opération est fastidieuse et chronophage.

L'objectif du projet est d'entraîner une intelligence artificielle à segmenter automatiquement les cellules de l'image.

## Current Business Process

La coupe histologique est un fichier de très haute résolution, qu'il n'est pas possible de traiter en une seule fois à l'écran. Par ailleurs, certaines zones de cette coupe ne doivent pas être prise en compte.

Les chercheurs commencent donc par extraire de cette image haute résolution des zooms sur des zones d'intérêts.

* Exemple : R22 VEGF 2_15.0xC1.jpg, R22 VEGF 2_15.0xC2.jpg, etc. sont des extraits d'une même coupe histologique R22
* Chaque zoom a une résolution 1920x1018 pixels et pèsent environ 2,5 Mo.

Dans Fiji, les chercheurs ouvrent chaque image zoom, et délimitent les cellules à la souris, ou à l'aide d'une tablette graphique. Chaque délimitation est enregistrée par Fiji sous la forme d'un fichier .roi (acronyme de "Region of interest").
* Il y a donc généralement plusieurs fichiers ROI pour chaque image zoom, et ils sont tous compressés dans un .zip
* Par exemple : RoiSetR22C1.zip contient sept ROI correspondant à des cellules dans l'image R22 VEGF 2_15.0xC1.jpg

Une fois les ROI générés, le logiciel est capable de calculer les surface occupées.

## Project Goal

L'objectif de ce projet d'IA est :
* de prendre en entrée une image zoom 1920x1018 pixels
* de produire en sortie
  * un fichier zip de ROI contenant les délimitations de chaque cellule dans l'image
  * une image avec les délimitations superposées

## Available Data

Pour des raisons de confidentialité et de taille de stockage, le repository github ne présente qu'un échantillon de données (dossier Data/sample).

L'ensemble des données est accessible aux contributeurs du projet sur un [drive google partagé](https://drive.google.com/open?id=1rmJG8g-bZpiiZyb6SJd3uqtqJOa-EQ9X)

* Dossier Images with ROI (545 Mo) :
  * 225 images 1920x1018
  * 210 ROI zip dans le sous-dossier Zip

NB.  225 <> 210 : Il y a une passe à faire pour retirer les images sans ROI

* Dossier Images without ROI (650 Mo):
  * 314 images 1920x1018
  * pas de ROI, mais ces images pourraient servir pour une phase d'apprentissage non supervisée (auto encoding)

Il sera possible d'avoir plus de données si le projet le nécessite (les chercheurs ont plusieurs Go de données de ce type)
