# Spike Sorting
This repository contains an object-oriented spike sorting implementation compatible with Scikit-learn architectures as `sklearn.pipeline.Pipeline` and `sklearn.model_selection.GridSearchCV`. The classes are defined in `spikesorting.py` and the notebook `Spikesorting_test.ipynb` shows how they can  be used. A Utah array recording for the test can be found in the dataset published by **Brochier, et al. 2018** [here](https://gin.g-node.org/INT/multielectrode_grasp). Additionally, the BlackRock library `brpylib.py` to read Utah array recordings can be downloaded [here](https://www.blackrockmicro.com/wp-content/software/brPY.zip). The file `Report.pdf` gathers a detailed description of the toolbox, the context in which it was developed, and practical results of its application in data from neurorehabilitation tasks of a stroke patient.

Please write me if you have any question understanding the code (guillemboada@hotmail.com).
## References
* Brochier, T. et al. *Data Descriptor: Massively parallel recordings in macaque motor cortex during an instructed delayed reach-to-grasp task*. Sci. Data 5, (2018). doi: 10.1038/sdata.2018.55.
