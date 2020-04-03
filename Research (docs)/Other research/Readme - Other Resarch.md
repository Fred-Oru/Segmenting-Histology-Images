## Other Research

Ce dossier contient les autres articles de recherche étudiés avant de faire le choix de Mask RCNN. Ils pourraient être utiles dans des phases ultérieures d'optimisation.


### Segmentation of Glomeruli using DL (2019)

Les auteurs passent un CNN avec une fenêtre glissante 300x300 sur des images 2560x1920 pour prédire pixel par pixel la chance d'appartenir à un glomérule (c'est un modèle plus simple que le Mask RCNN mais probablement très long à exécuter ...)

Ils partent de 250 images d'histologie 2560x1920x3 qu'ils découpent en patch de 300x300x3 avec un stride de 20 pixel pour nourrir un CNN.

Chaque patch est labellisé (NPS)'glomérule sain ou presque' ou (GS) 'glomérule pathologique' ou 'pas de glomérule'.

CNN utilisé : Google Inception V3 pretrained sur Imagenet, ré-Entraîné pour reconnaître l'une des trois classes => 90-95% d'accuracy

Train / valid set : ils ont mis 70% des patients dans un jeu de train, et 30% dans le valid, et *ensuite* ils ont fait les cropping. Il est important que des images de patients ne soient pas à la fois dans le train et le test

Ils ont fait une cross validation à 4 fold pour bien estimer la capacité du modèle. Ils ont fait de la data augmentation (copies avec différentes white noise)

### Semantic Edge detection

Cet article pourrait être intéressant pour faire un pré-traitement des images en éliminant les couleurs pour ne garder que les bords avant de passer dans Mask R-CNN

Il s'agit d'un détecteur de bord qui s'appuie sur le modèle Casenet : le principe est d'utiliser un Resnet et d'extraire les features à plusieurs endroits du réseaux pour en faire une convolution à part et en déduire les bords. Dans l'article ils vont un peu plus loin en entraînant différents les couches profondes et les couches hautes.

### Segmentation Classification HoVer-Net.pdf

La complexité de l'article tient à leur problématique spécifique de l'occlusion des noyaux : leur problème est de bien séparer les bords des noyaux, même quand ils sont superposés. Mais pour notre cas, il est intéressant de regarder la Fig2. branche NP : ils arrivent à faire un masque de segmentation en utilisant
* un Resnet50
* une suite de upsampling / convolution / dense layer
* des skip connections (pas précisé où)

Il serait peut être possible d'utiliser cette structure, moins lourde qu'un Mask RCNN, pour résoudre notre problème

code : https://github.com/vqdang/hover_net
tensorflow V1, pas simple à suivre ...

### Nature Deep Learning Histology

Titre original : Automated acquisition of explainable knowledge from unannotated histopathology images

Cet article n'est pas directement en phase avec notre problématique. Ils font une annotation automatique des zones pathologiques d'une image histologique, sachant si l'image correspond à un patient sain ou malade.

Ce qui est intéressant est qu'ils utilisent un auto-encodeur sur les images de haute résolution pour les réduire à 100 features numériques, sur lequels ils appliquent un apprentissage supervisé classique.

Leçon à retenir : l'auto-encoding pourrait être un moyen de réduire la dimensionnalité des images tout en gardant les éléments importants. Peut être peut on forcer l'auto encodeur à ne conserver que les silhouettes de glomérules ?
