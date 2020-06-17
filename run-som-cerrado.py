import csv
from gsom import init_grid
from gsom import start_som
from gsom import print_clustering
from gsom import plot_map
from gsom import get_neuron_labels
from gsom import check_neuron_accuracy
from gsom import generate_confusion_matrix

data_file = "/Users/rodrigos/Documents/INPE/GSOM-python/cerrado-nofilter-data.csv"
labels_file = "/Users/rodrigos/Documents/INPE/GSOM-python/cerrado-nofilter-labels.csv"

# load the input data
input = []

with open(data_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ',')
    for row in csv_reader:
        input.append(row)
        
# cast all the input to float
for i in range(len(input)):
    for j in range(len(input[i])):
        input[i][j] = float (input[i][j])
        
# load the input labels data
input_labels = []

with open(labels_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ',')
    for row in csv_reader:
        input_labels.append(row[0])

init_grid (input, 25, 25, sf=1.0, alfa=1.0)

#start_growing_phase (input, 10)

start_som (input, 1)

print_clustering(input)

neuron_labels = get_neuron_labels(input, input_labels)

check_neuron_accuracy(input, neuron_labels, input_labels)

generate_confusion_matrix(input, neuron_labels, input_labels)

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


