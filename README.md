# Quantum Machine Learning Project

This repository contains the code and report for the Quantum Machine Learning (QML) project, focused on solving the Parity Problem using various quantum circuits and classical neural networks. The project was developed as part of the QML course at the University of Athens.

## Project Overview

### The Parity Problem

The parity problem involves determining whether the number of ones in a binary string is even or odd. In this project, we consider two versions of the problem: one with 3 inputs and another with 5 inputs.

### Data Description

- **Training Data:** Binary strings with corresponding parity labels, with added noise to simulate real-world scenarios. Files: `classA_train.dat`, `classB_train.dat`
- **Test Data:** Binary strings with corresponding parity labels used to evaluate the classifier. Files: `classA_test.dat`, `classB_test.dat`

### Quantum Circuits

#### Circuit (a)

- **Design:** Utilizes 3 qubits with angle embedding and basic entangler layers.
- **Cost Function:** Mean Squared Error (MSE)
- **Optimization Method:** Nesterov Momentum Optimizer

![](./results/circuit_a.png)

#### Circuit (b)

- **Design:** Alternative ansatz circuit with different gate arrangements.
- **Cost Function:** Mean Squared Error (MSE)
- **Optimization Method:** Nesterov Momentum Optimizer

![](./results/circuit-ansatz.png)

#### Circuit (c)

- **Design:** Utilizes data re-uploading technique with multiple re-uploading layers.
- **Cost Function:** Mean Squared Error (MSE)
- **Optimization Method:** Adam Optimizer

![](./results/circuit-c.png)

### Evaluation

The classifiers' performance is evaluated based on accuracy on both training and test datasets. Various optimizers were tested, including Adam, Gradient Descent, Nesterov Momentum, and RMSProp.

### The Parity Problem with 5 Inputs

A more complex version of the parity problem was tackled using a quantum circuit with 5 qubits and data re-uploading technique.

![](./results/bonus-circuit.png)

### Comparison with Classical Neural Network

A classical neural network was trained on the same dataset to benchmark the performance of the quantum classifiers.

## Deliverables

- **Code Notebooks:**
  - `qml_classifier_a_model.ipynb`: Circuit (a) implementation and results
  - `qml_classifier_b_experiments_circuits.ipynb`: Circuit (a), (b), (c) implementation and results
  - `qml_classifier_b_best_model.ipynb`: Circuit (c) implementation and results, optimizers survey code and results
  - `qml_classifier_c.ipynb`: Circuit for Parity-5 implementation and results

## References

- PennyLane Variational Classifier Tutorial: [link](https://pennylane.ai/qml/demos/tutorial_variational_classifier/)
- PennyLane Data Re-uploading Classifier Tutorial: [link](https://pennylane.ai/qml/demos/tutorial_data_reuploading_classifier/)
- B.Sc. Thesis: [link](https://pergamos.lib.uoa.gr/uoa/dl/object/3393917/file.pdf)

## Author

Konstantinos Nikoletos
