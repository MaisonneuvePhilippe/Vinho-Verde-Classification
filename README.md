Vinho Verde Classification code for the course INF8215

For this assignment, we took part in an Inclass Kaggle challenge whose goal is to develop a learning approach
automatic (ML) to measure the quality of a wine between 1 and 10.

Our model seems to have succeeded in making excellent predictions on the level of wine quality.
Portuguese. In fact, we have achieved an accuracy of over 71% using a forest
decision tree of type extra trees whose parameters have been optimized using research
random followed by a grid search. Based on the dataset provided to us,
we feed our model with these data combined at a polynomial level 2 and reduced to the
dimension 28 using a PCA.

We believe that other methods might offer interesting results and would have liked
further study the possibility of using gradient reinforcement trees, because depending on the
literature, they appear to be able to outperform decision tree forests in most
case. The AdaBoost model also seemed to yield some interesting results.
Finally, the combination of these different models could also give a good result, if they
don't all make the same kind of mistake.

