# Model Overview
This repository contains the code for SwinCross: Cross-modal Swin Transformer for 3D Medical Image Segmentation. 

### Installing Dependencies
First install uv to be able to deal with different python versions. 

Then create a venv using : 
``` bash
uv venv name_of_env --python 3.12
```

Dependencies can be installed using:
``` bash
uv pip install -r requirements.txt
```

### Preparing the dataset 
Le fichier ‘dataset_builder_simpleTK.py’ a pour rôle de construire le dataset à partir des images du dossier ‘Task_1_15examples’. Donc, en le lançant (aucun paramètre requis), vous construisez un dossier ‘Dataset_Final_SwinCross_SITK‘ à la racine qui contiendra le fichier ‘dataset_swincross.json’. Ce fichier a 12 données d’entraînement pour 3 de validation, combinaisons différentes à chaque fois grâce au shuffle.

### Training
Le fichier ‘train.py’ peut être lancé directement puisque nous lui avons mis, en paramètres par défaut ce qu’il faut. Ainsi s’il est lancé directement sans paramètres :
- Le paramètre --logdir (répertoire des logs et fichiers de sauvegarde) occasionnera par défaut la création d’un répertoire ‘runs’ qui aura un répertoire ‘for_log’ comme fils.
- Le paramètre --checkpoint est à None par défaut (il ne fait aucune reprise).  Mais si jamais des checkpoints existent, ils seront tous les deux dans le répertoire ‘runs/for_log’. Et il y aura deux checkpoints : Celui qui ressence les hyperparamètres qui ont donné la meilleure accuracy jusque là, donc le meilleur modèle (model_best.pth), et celui qui ressence juste le dernier modèle (model_last.pth)
Donc si jamais vous voulez reprendre un entraînement interrompu, pour le checkpoint vous fournissez un de ces deux fichiers selon vos besoins. S’il y ‘en a pas, vous pouvez laisser –checkpoint à None.
 
- Le paramètre --pretrained_dir qui est le répertoire où seront sauvegardés les poids pré-entraînés pour du fine_tuning, mais comme on n’a pas ça, on ne peut pas faire de fine tuning, donc c’est mis à None par défaut
- Le param --data_dir qui est le dossier ‘Dataset_Final_SwinCross_SITK’ (dossier du dataset généré par ‘dataset_builder_simpleTK.py’) par défaut
- Le param  --json_list est par défaut ‘dataset_swincross.json’ le json généré par ‘dataset_builder_simpleTK.py’.
--pretrained_model_name à None encore une fois puisque nous n’avons pas de poids préentrainés

Donc train.py peut être lancé directement sans paramètres. Sauf si vous voulez mettre d’autres paramètres pour des tests ou autres. En l’état, par défaut sinon aussi, c’est 3000 époques, 50 époques de warmum, taille de batch_size de 6, etc.

Après, pour du petit test en local, peut-être des paramètres basiques peuvent être utilisés : 
``` bash
‘#CUDA_LAUNCH_BLOCKING=1 python 3.12 train.py --batch_size 2 --cache_rate 0.0 --max_epochs 2 --val_every 1 --workers 0 --logdir test_debug
 ```

A SwinCross network with standard hyper-parameters for the task of head and neck tumor semantic segmentation (HECTOR dataset) can be defined as follows:
``` bash
model = SwinCross(
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 2
    config.in_chans = 2
    config.embed_dim = 48  # change 128 or 192
    config.depths = (2, 4, 2, 2)  # change 4 to 6,10
    config.num_heads = (3, 6, 12, 24)
    config.window_size = (3, 3, 3)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = True
    config.spe = False
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.seg_head_chan = config.embed_dim // 2
    config.num_classes = 3  # Modification : added num_classes parameter in order to pass 
                            #the number of labels expected in the ouput segmentation map to the model
    config.img_size = (96, 96, 96) # Important to keep this size for know for  patches to be divisible by 32 (downsample) and by 3 (window size) 
    config.pos_embed_method = 'relative'
    return config
```

To initiate distributed multi-gpu training, ```--distributed``` needs to be added to the training command.

To disable AMP, ```--noamp``` needs to be added to the training command.


### Testing
Le fichier test.py, nous lui avons également mis tout ce qu’il faut par défaut et il peut donc être lancé immédiatement après l’entraînement sans paramètres.
Par exemple, il prend en paramètres un param --pretrained_dir (cette fois ici pretrained_dir est le répertoire des poids à utiliser pour l’inférence. Nous lui avons donc donné le dossier ‘runs/for_log’),  --pretrained_model_name (le fichier des poids à utiliser pour l’inférence : nous avons mis par défaut le fichier qui garde les checkpoints du meilleur modèle : ‘model_best.pth’)

Dans le cas où le petit test en local a été effectué,il faut modifier le dossier de référence du paramètre --pretrained_dir vers 'runs/test_debug' comme précisé auparavant.
``` bash
python3.12 test.py --pretrained_dir runs/test_debug
```
You can use the state-of-the-art pre-trained TorchScript model or checkpoint of UNETR to test it on your own data.

Once the pretrained weights are downloaded, using the links above, please place the TorchScript model in the following directory or 
use ```--pretrained_dir``` to provide the address of where the model is placed:

```./pretrained_models``` 

The following command runs inference using the provided checkpoint:
``` bash
python test.py
--infer_overlap=0.5
--data_dir=/dataset/dataset0/
--pretrained_dir='./pretrained_models/'
--saved_checkpoint=ckpt
``` 

Note that ```--infer_overlap``` determines the overlap between the sliding window patches. A higher value typically results in more accurate segmentation outputs but with the cost of longer inference time.

If you would like to use the pretrained TorchScript model, ```--saved_checkpoint=torchscript``` should be used.


