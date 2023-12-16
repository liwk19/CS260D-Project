### Coreset Selection for Few-shot Transfer Learning in High-Level Synthesis Simulation

This repository is for the course project of CS260D. It is adapted from the repository of our research project https://github.com/yunshengb/software-gnn.git, "weikai" branch. Since the research project contains a lot of necessary codes related to high-level synthesis and compilation, it is very complex and large. You can only focus on the following files that are related to this course project:

- src/config.py: hyper-parameter configurations and settings

- src/coreset.py: coreset selection methods

- src/data.py: it contains some codes that are also relevant to coreset selection

- src/train.py: it contains model ensembling methods

- src/hidden_save/similarity.py: visualization

  

Here we introduce how to reproduce our results by simply changing the settings in src/config.py. All the settings below should be done in src/config.py

##### 1, Pre-train one model or multiple models

If you want to pre-train one model on all training kernels, please activate lines 40-41 and deactivate line 42; activate lines 58-63 and deactivate line 66; set transfer_learning in line 430 to False. Then running "python src/main.py" can pre-train the model for 1,500 epochs. Please remember the printed location where the model is saved. After pre-training, you need to set model_path in line 327 to the model location, so that you can use the pre-trained model.

If you want to pre-train one model on each training kernel so that you can get 33 models, please deactivate lines 40-41 and activate line 42; deactivate lines 58-63 and activate line 66; change line 66 to the 33 kernels one by one (the 33 kernels' names are in "ensemble_KERNEL" in line 439~443) and run "python src/main.py". In this case, you do not need to care about the model saving location, because the location is fixed.

##### 2, Fine-tune the one model pre-trained on all training kernels

Please deactivate lines 40-41 and activate line 42; deactivate lines 58-63 and activate line 66, and change the kernel name in line 66 to your desired test kernel (you can only use one test kernel at a time). Please set transfer_learning in line 430 to True; set "--coreset" to "random" to utilize random selection, or "CRAIG" to utilize coreset selection. If you select "CRAIG", set "--craig_metric" to "h5~h7" to use the hidden layer's representation, "h8" to use the model output, or "uncertainty" to use uncertainty. If you select "uncertainty", please further set "--uncertainty_metric" to "high_uncertain" or "low_uncertain".

##### 3, Model ensemble

Please deactivate lines 40-41 and activate line 42; deactivate lines 58-63 and activate line 66, and change the kernel name in line 66 to your desired test kernel (you can only use one test kernel at a time). Please set transfer_learning in line 430 to True; set "--coreset" to "ensemble_train"; set "--uncertainty_metric" to "high_uncertain" or "low_uncertain"; change the root directory in "ensemble_model_path" to your root directory. Then you can run "python src/main.py" to fine-tune the 33 models. After fine-tuning, set "--coreset" to "ensemble_inference"; set "--ensemble_metric" to "average", "lowest_loss", or "weighted".

Please feel free to contact us if there is any confusing point.
