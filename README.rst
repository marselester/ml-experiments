============================
Machine Learning Experiments
============================

For this programming exercise I will implement a linear regression with
multiple variables to predict **concrete compressive strength**.
Concrete is the most important material in civil engineering.
The concrete compressive strength is a function of age and ingredients,
such as cement and water. The dataset was found at
`The UCI Machine Learning Repository`_, donated by professor I-Cheng Yeh
[#DatasetOwner]_.

`The dataset`_ consists of eight features (input variables) measured on
1030 experiments. First, let's list features:

1. cement
#. blast furnace slag
#. fly ash
#. water
#. superplasticizer
#. coarse aggregate
#. fine aggregate
#. age

All of them are in :math:`kg/m^3` except for age, that is in days (1–365).
The target variable (concrete compressive strength) is in :math:`MPa`
(megapascale). Data is in raw form (not scaled).

Prepare The Dataset
-------------------

We should split the dataset to three parts:

- training set 60%
- cross validation set 20%
- test set 20%

It will help us to test algorithm's performance. After that we need to apply
feature scaling and mean normalization. They are needed to speed up
a gradient descent (fewer steps to converge).

Feature scaling makes sure that features are on a similar scale,
approximately in :math:`–1 \le x_i \le 1` range. Mean normalization replaces
:math:`x_i` with :math:`x_i - \mu_i` to make features have approximately
zero mean. It is important to get **mean** and **standard deviation** of
training set and normalize all sets using those parameters.

.. _The UCI Machine Learning Repository: http://archive.ics.uci.edu/ml/
.. [#DatasetOwner] I-Cheng Yeh, "Modeling of strength of high performance
   concrete using artificial neural networks",
   Cement and Concrete Research, Vol. 28, No. 12, pp. 1797-1808 (1998)
.. _The dataset: https://github.com/marselester/ml-experiments/blob/master/data/concrete.csv
