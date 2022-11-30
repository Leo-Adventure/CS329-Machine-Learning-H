![Screen Shot 2022-11-24 at 14.41.02](/Users/leo/Desktop/Screen Shot 2022-11-24 at 14.41.02.png)

1. SVM can not be used for unsupervised clustering, because it needs the true label to construct a split plane. And, SVM can not be used for data dimension reduction, because it usually needs to level up the dimension of the dataset to achieve a better plane for splitting.

2. The strengths of SVM are:

   - The precision of classification is high and the ability to generalization is great.
   - There are many kernel functions to use, so it can handle many non-linear problems.

   When the amount of dataset is not large, the SVM usually performs well.

3. The weaknessed of SVMs are:

   - Really sensible of the absence of data.
   - Hard to choose a kernel function from various functions.

   When the amount of dataset is large, the SVM usually performs badly.

4. When the amount of dataset is not large, there are not too many absent data, and the dimension of data is relatively high, the SVM is a good candidate for the classification.