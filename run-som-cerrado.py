import csv
import GSOM

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

GSOM.init_grid (input, 25, 25, sf=1.0, alfa=1.0)

GSOM.start_som (input, 1)

GSOM.print_clustering(input)

neuron_labels = GSOM.get_neuron_labels(input, input_labels)

GSOM.check_neuron_accuracy(input, neuron_labels, input_labels)

GSOM.generate_confusion_matrix(input, neuron_labels, input_labels)

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

GSOM.plot_map(input, input_labels, neuron_labels, labels_colors, show_samples = False)


