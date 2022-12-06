## Approach Summary
* The images are aggregated as a dataset by making its size uniform of 256*256 and normalizing its pixel intensity.
* For prepare_train_data: Dataset is sampled into train and validation split from given data.
* Class Weights are derived inversely proportional to number of instances, to infer weight of each instance. Weighted Sampler using the aforementioned weights is used to sample the batches in trainloader.
* Model: MyModel Class contains the proposed model architecture with a stack of Convolution, MaxPool, Convolution and 3-head linear classifier head.
* Reason: CNN proven to be a baseline approach, hence applied to set a valid benchmark.

## Specifications:
* Device: CPU
* Run On: 8 GB Mac M1 Chip
* IDE Used: PyCharm, Colab (for initial experimentation)

## Limitations: 
* Performance Improvement Scope restricted owing to low number of instances.

## Improvements/Possible Approach:
1. Transfer learning via a deep-trained model on similar task.
2. Triplet Loss Learning could be something to explore: the skin diseases seem similar in lot many cases and even instances of herpes are less.
Therefore, sampling anchor, positive and negative pair batches for similar classes - we can make the model learn to `maximise loss in terms of its difference with other diseases` rather than cross entropy loss which just penalizes the model for wrong classification.


Test for unseen data, section included in README.md

