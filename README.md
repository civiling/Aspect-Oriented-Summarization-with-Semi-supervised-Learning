# Aspect Oriented Summarization with Semi-supervised Learning

## Proposed method
First, train a model with labeled data to create supervised data.

Next, the unlabeled data is used for back translation to create noise injected unlabeled data.

Next, train the model with the unlabeled data and noise injected unlabeled data together to create unsupervised data.

Finally, aspects are extracted from supervised and unsupervised data and aspect-oriented summarization is performed.

