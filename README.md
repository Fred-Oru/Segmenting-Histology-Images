# Glomeruli Segmentation in Histology images

This repository is an Artificial Intelligence project aiming to segment [kidney glomeruli](https://en.wikipedia.org/wiki/Glomerulus_(kidney)) in histology images.

## Business Problem

Des chercheurs en biologie étudient les effets d'une pathologie sur les [glomérules](https://fr.wikipedia.org/wiki/Glom%C3%A9rule_r%C3%A9nal) du rein. Pour mesurer ces effets, ils étudient des coupes histologiques de reins sur lesquels la pathologie est révélée par un marquage de couleur qui ressort généralement en marron foncé sur un fond marron clair. Le marquage bleu correspond à des noyaux de cellules. Pour quantifier le niveau de pathologie du rein, les chercheurs doivent estimer le pourcentage de la surface des glomérules qui est marquée en marron foncé.

A l'aide d'un logiciel de gestion d'images (Distribution [Fiji](http://fiji.sc/) du logiciel open source [Imaji](https://en.wikipedia.org/wiki/ImageJ), les chercheurs :
* délimitent manuellement les contours des glomérules précisément la membrane basale de la caspule de Bowman),
* éliminent les marqueurs bleus des noyaux de cellules, et font un seuillage pour ne garder que la surface marquée en marron foncé
* mesure à l'aide du logiciel la surface marquée et la surface totale du glomérule pour en déduire le pourcentage marqué.

L'opération la plus fastidieuse et chronophage est celle qui consiste à délimiter manuellement les glomérules à l'aide d'une souris ou d'une tablette graphique.

L'objectif du projet est d'entraîner une intelligence artificielle à segmenter automatiquement les glomérules dans une image histologique de rein.

## Data specifications

La coupe histologique numérisée est livrée aux chercheurs sous la forme d'un fichier de très haute résolution, dans un format propriétaire qui ne peut être visualisé qu'avec un logiciel spécifique. Pour des raisons de taille et de format, il n'est pas possible de procéder aux mesures directement sur cette image. En revanche, il est possible de sélectionner une zone et l'exporter au format jpg. Par ailleurs, les chercheurs ne s'intéressent qu'à certaines zones de l'image, là où la coupe est nette et montre une section significative des glomérules.

Pour toutes ces raisons, les chercheurs commencent par extraire de l'image haute résolution des zooms sur des zones d'intérêts. Par exemple (cf. dossier Data/sample) :
*  les fichiers R22 VEGF 2_15.0xC1.jpg à R22 VEGF 2_15.0xC5.jpg sont extraits d'une même coupe histologique du rein n° R22
* un zoom a typiquement une résolution 1920x1018 pixels et pèse environ 2,5 Mo. Il peut cependant y avoir des résolutions légèrement différentes.

NB sur la nomenclature des fichiers :
* VEGF est un des types de marqueurs utilisés. D'autres coupes sont effectuées avec des reins marqués par PAS ou HES.
* 15 est le grossissement du zoom par rapport à l'image haute définition d'origine.

Dans Fiji, les chercheurs ouvrent chaque image zoom, et délimitent les cellules à la souris, ou à l'aide d'une tablette graphique. Chaque délimitation est enregistrée par Fiji sous la forme d'un fichier .roi (acronyme de "Region of interest"). Il y a donc généralement plusieurs fichiers ROI pour chaque image zoom, et ils sont tous compressés dans un .zip
* Par exemple : RoiSetR22C1.zip contient sept ROI correspondant à des cellules dans l'image R22 VEGF 2_15.0xC1.jpg
* Chaque ROI contient (entre autres) les coordonnées (x,y) de chaque pixel du contour délimité par le chercheur.

Une fois les ROI générés, le logiciel est capable de calculer la surface du glomérule et la surface marquée à l'intérieur du glomérule.

## Project Goal

L'objectif de ce projet d'intelligence artificielle est :
* de prendre en entrée une image zoom
* de produire en sortie
  * un fichier zip de ROI contenant les délimitations de chaque cellule dans l'image
  * une image avec les délimitations superposées

## Available Data

Pour des raisons de confidentialité et de taille de stockage, le repository github ne présente qu'un échantillon de données (dossier Data/sample).

L'ensemble des données est accessible aux contributeurs du projet sur un [drive google partagé](https://drive.google.com/open?id=1rmJG8g-bZpiiZyb6SJd3uqtqJOa-EQ9X). Il sera possible de récupérer plus de données de ce type si le projet le nécessite (les chercheurs en ont plusieurs Go).

* Dossier glomerulus/train (500 Mo):
  * 200 images de reins pathologiques et de reins sains
  * résolution 1920x1018  en général, quelquefois 1018x1920


* Dossier Images with ROI (545 Mo) :
  * 200 ROI zip dans le sous-dossier Zip
  * les images de rein étaient dans ce dossier à l'orgine, mais ont été transférés sur glomerulus/train après traitement


* Dossier Images without ROI (650 Mo):
  * 314 images 1920x1018 (vérifier si la résolution est constante)
  * ici le marqueur violet est juste un colorant, il ne révèle pas les pathologies. Ces images sont utilisées pour calculer la surface blanche à l'intérieur des glomérules, afin de vérifier si leur structure est normale.
  * Nous n'avons pas les ROI car ces fichiers n'ont pas été étudiés. Cela étant ces images pourraient servir pour une phase d'apprentissage non supervisée (auto encoding), ou de test.
