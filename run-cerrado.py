import csv
from gsom import init_gsom
from gsom import start_growing_phase
from gsom import start_smoothing_phase
from gsom import print_classification
from gsom import plot_scatter
from gsom import label_neurons
from gsom import check_gsom_accuracy
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

######################

init_gsom (input, sf=1.0, alfa=1.0)

start_growing_phase (input, 1)

start_smoothing_phase (input, 1)

print_classification(input)

neuron_labels = label_neurons(input, input_labels)

check_gsom_accuracy(input, neuron_labels, input_labels)

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

plot_scatter(input, input_labels, neuron_labels, labels_colors, show_samples = False)


