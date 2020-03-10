## Articles les plus intéressants pour notre cas
vxcvv

###  Mask R-CNN.pdf:
L'article est touffu est s'appuie sur la connaissance des modèles qui ont précédé Mask R-CNN : R-CNN / Fast R-CNN et Faster R-CNN.

Mask RCNN est composé d'un :
* RPN (region proposal Network) qui propose des bounding boxes candidates sur l'image, et qui s'appuie généralement sur des réseaux (pré-entraînés ?) Resnet
* d'un réseau qui prédit (pour chaque boite candidate si j'ai bien compris) la classe, et un offset sur une boite
* *en parallèle* un réseau qui prédit un masque binaire pixel par pixel

L'ensemble est entraîné d'un coup avec une loss qui somme les loss de chaque opération.

here is a keras implementation
https://github.com/matterport/Mask_RCNN#segmenting-nuclei-in-microscopy-images-built-for-the-2018-data-science-bowl

NB : dans notre cas on a qu'un classe à apprendre (glomérule), le cadre peut être intéressant pour produire des patches, mais c'est surtout la surface qui nous intéresse.

Nb : on parle à un moment de ROIPooling des boites candidates. Il s'agit d'une opération de max pooling qui permet de transformer chaque bounding box (de tailles a priori différentes) en feature map de taille fixe.

### Segmentation of Glomeruli using DL (2019)

Semble parfaitement en ligne avec notre recherche, à un détail près : ils cherchent à distinguer les glomérules sains et les glomérules pathologiques, alors que nous n'avons que des glomérules sains qui contiennent en leur sein des noyaux pathologiques. Cela étant, la méthode devrait être transposable sur la segmenation.

Ils partent de 250 images d'histology 2560x1920x3 (cohérent avec notre propre jeu de données) qu'ils découpent en patch de 300x300x3 avec un stride de 20 pixel pour nourrir un CNN. Chaque patch est labellisé (NPS)'glomérule sain ou presque' ou (GS) 'glomérule pathologique' ou 'pas de glomérule'. Ensuite ils passent le CNN avec une fenêtre glissante 300x300 sur la totalité d'une image 2560x1920 pour prédire pixel par pixel la chance d'appartenir à un glomérule. C'est de ceal qu'on extrait la segmentation (en choisissant un seuil). C'est en fait la méthode qui était utiliseé avant les FCN comme Yolo ! Faisable mais probablement très long ...

CNN utilisé : Google Inception V3 pretrained sur Imagenet
Ré-Entraîné pour reconnaître l'une des trois classes
90-95% d'accuracy

Train / valid set : ils ont mis 70% des patients dans un jeu de train, et 30% dans le valid, et *ensuite* ils ont fait les cropping. Important que des images de patients ne soient pas à la fois dans le train et le test

Ils ont fait une cross validation à 4 fold pour bien estimer la capacité du modèle
Ils ont fait de la dataugmentation (copeis avec différentes white noise)

## Articles développant d'autres techniques

### Deep learning for digital pathology image analysis

CNN network : AlexNet, trained on CIFAR‑10 (10 couches, input 32x32).
Optimizer = AdaGrad, 22H d'entraînement sur un GPU
Hyperparameters -> page 7

Le modèle est utilisé comme classificateur sur un patch (petite partie d'une image)
Lien technique associé : http://www.andrewjanowczyk.com/use-case-3-tubule-segmentation/
ground truth : dans notre cas il faudrait probablement générer les images avec le masque de surface (ie les ROI avec leur surface intérieure)
Pour chaque image il extrait au hasard des patches
A la fin de réseau produit pour chaque pixel un probabilité d'être ou ne pas être dans un tubule
Strucrure du modele : https://github.com/choosehappy/public/blob/master/DL%20tutorial%20Code/common/BASE-alexnet_traing_32w_dropout_db.prototxt

### Nature Deep Learning Histology

Titre original : Automated acquisition of explainable knowledge from unannotated histopathology images

Cet article n'est pas directement en phase avec notre problématique. Ils font une annotation automatique des zones pathologiques d'une image histologique, sachant si l'image correspond à un patient sain ou malade.

Ce qui est intéressant est qu'ils utilisent un auto-encodeur sur les images de haute résolution pour les réduire à 100 features numériques, sur lequels ils appliquent un apprentissage supervisé classique.

Leçon à retenir : l'auto-encoding pourrait être un moyen de réduire la dimensionnalité des images tout en gardant les éléments importants. Peut être peut on forcer l'auto encodeur à ne conserver que les silhouettes de glomérules ?

# Segmentation Classification HoVer-Net.pdf

La complexité de l'article tient à leur problématique spécifique de l'occlusion des noyaux : leur problème est de bien séparer les bords des noyaux, même quand ils sont superposés. Mais pour notre cas, il est intéressant de regarder la Fig2. branche NP : ils arrivent à faire un masque de segmentation en utilisant
* un Resnet50
* une suite de upsampling / convolution / dense layer
* des skip connections (pas précisé où)

code : https://github.com/vqdang/hover_net
tensorflow V1, pas simple à suivre ...

### CNN for Edge Detection (bof)

Une approche pour trouver les glomérules serait d'afficher les bords dans l'image. Une technique assez simple est de faire une simple convolution avec des noyaux de Sobel, comme semble le faire [ImageJ -Find Edges](https://imagej.nih.gov/ij/docs/guide/146-29.html).

Partant de cette idée, j'ai cherché des articles de recherche traitant de la détection de bord. J'ai trouvé celui-ci, qui utilise les réseaux CNN et les compare à des noyaux classiques comme (Sobel, Canny, LOG) ou d'autres méthodes comme Tsallis, montrant que les détectent plus de bords, mais on souvent un moins bon ratio bruit sur signal.

### Semantic Edge detection (bof)

Un détecteur de bord qui s'appuie sur Casenet : le principe est d'utiliser un Resnet et d'extraire les features à plusieurs endroits du réseaux pour en faire une convolutin à part et en déduire les bords. Dans l'article ils vont un peu plus loin en entraînant différents les couches profondes et les couches hautes.
