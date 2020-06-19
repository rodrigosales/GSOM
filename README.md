# GSOM Python Package

The package allows two different input methods, the first one starts a GSOM execution, with both Growing phase and Smoothing phase. On the above example, a GSOM grid is initialized with a Spread Factor configured as 1.0 and a 0.7 Learning Rate. After that, a Growing Phase with 5 iterations will be started, and followed by a Smoothing Phase with 15 iterations. 

```python
init_grid (input, sf = 1.0, alfa = 0.7)
start_growing_phase (input, 5)
start_smoothing_phase (input, 15)
```

As the SOM algorithm is very similar with a GSOM Smoothing phase, an alias for SOM execution is also provided as a second input method. In order to start a simple SOM, we need to provide the grid size and a Learning Rate. In the following example, a 25x25 grid is initialized with a 0.6 Learning Rate, and 15 iterations will run.

```python
init_grid (input, 25, 25, 0.6)
start_som (input, 15)
```

We provide 5 output methods, to be able to retrieve GSOM results. The first one prints the best matching unit of each sample, indicating the neuron that best cluster each sample.

```python
print_clustering(input)
```

In order to provide a classification based on the generated clusters, a method was developed to label a neuron, based on the most frequent samples associated to each neuron, this output can be retrieved as:

```python
neuron_labels = get_neuron_labels(input, input_labels)
```

Based on the input labels, and the retrieved neuron labels, its possible to get the accuracy for each neuron:

```python
check_neuron_accuracy(input, neuron_labels, input_labels)
```

Based on the same data, we are able to generate the confusion matrix, and get the classification accuracy for each class:

```python
generate_confusion_matrix(input, neuron_labels, input_labels)
```

Finally, providing e color for each label, we are able to plot a scatter and visually check the network topology:

```python
labels_colors = {
    "Pasture":"#F77B01",
    "Cerrado":"#2B6490",
    "Forest":"#4EAF4A",
    "Fallow_Cotton":"#E41B1D",
    "Soy_Fallow":"#C26596",
    "Soy_Corn":"#BFC127",
    "Soy_Sunflower":"#CCEBC5",
    "Soy_Millet":"#949494",
    "Soy_Cotton":"#A65628"
}

plot_map(input, input_labels, neuron_labels, labels_colors, show_samples = False)
```

