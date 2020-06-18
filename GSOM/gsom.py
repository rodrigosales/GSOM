import random
import array
import time
import matplotlib.pyplot as plt
from math import exp
from math import sqrt
from math import log

def init_grid (input, initial_width = 2, initial_height = 2, sf = 0.3, alfa = 1.0):
    
    samples_size = len(input[0])
    
    global weights
    weights = []
    
    global coordinate_map
    coordinate_map = []
    
    global acumulated_error
    acumulated_error = []  
    
    global gt
    gt = -samples_size * log(sf)
    
    global fd
    fd = 0.5
    
    global initial_learning_rate
    initial_learning_rate = alfa
    
    total_neurons = initial_width * initial_height;
    
    # create the weights vector
    for i in range (total_neurons):
        temp = []
        
        for j in range (samples_size):
            temp.append(random.random())
        
        weights.append(temp)
        
    # create the coordinate map
    for i in range (initial_width):
        for j in range (initial_height):
            coordinate_map.append([i, -1 * j])
            
    # create the acumulated error vector
    for i in range (total_neurons):
          acumulated_error.append(0)

    return
    
def start_growing_phase (input, epochs):
    
    print ("-----------------------")
    print ("Growing Phase")
    print ("-----------------------")
    
    # for each epoch
    for i in range (epochs):
        
        ts = time.time()
        
        print ("Epoch: " + str(i))
  
        # show a random sample to the network
        #sample = random.choice(input)
        
        # show all the samples
        for j in range(len(input)):
        
            sample = input[j]
            
            # find the best matching unit
            bmu = find_best_matching_unit(sample)
            winner = bmu["winner"]
            error =  bmu["error"]

            # update the winner and the neighborhood
            update_neighborhood(winner, sample, i, epochs)
        
            # save the acumulated error vector
            acumulated_error[winner] = acumulated_error[winner] + error
            #print (acumulated_error)
        
            # check if the acumulated error is greater then the GT
            if (acumulated_error[winner] > gt):
            
                boundary = check_boundary (winner, "available")
            
                if not boundary:
                    spread_error(winner)
                else:
                    for b in boundary:
                        grow (winner, b)
                        acumulated_error[winner] = 0
        print("time:" + str(time.time() - ts) + " seconds")
    
    return
    
def start_som (input, epochs):
    start_smoothing_phase (input, epochs)
    return
    
def start_smoothing_phase (input, epochs):

    print ("-----------------------")
    print ("Smoothing Phase")
    print ("-----------------------")
    
    # for each epoch
    for i in range (epochs):
        
        ts = time.time()
        
        print ("Epoch: " + str(i))
  
        # show a random sample to the network
        #sample = random.choice(input)
        
        # show all the samples
        for j in range(len(input)):
        
            sample = input[j]
            
            # find the best matching unit
            bmu = find_best_matching_unit(sample)
            winner = bmu["winner"]
            error =  bmu["error"]

            # update the winner and the neighborhood
            update_neighborhood(winner, sample, i, epochs)
            
        print("time:" + str(time.time() - ts) + " seconds")
    
    return
    
def find_best_matching_unit (sample):
    
    result = dict()
    
    min_error = float('inf')
    winner = 0;

    # for each neuron weight
    for i in range(len(weights)):
        
        # for each weight
        d = 0
        for j in range(len(weights[i])):
            
            # calculate the distance between the weight and the sample
            d = d + pow(weights[i][j] - sample[j], 2)
            
        d = sqrt(d)    
       
        
        # find the minimum error  
        if (d < min_error):
            min_error = d
            winner = i

    #print ("winner: " + str(winner))
    
    result["winner"] = winner
    result["error"] = min_error
    
    # return the winner
    return result
    
def update_neighborhood(winner, sample, iteration, epochs):
    
    #print("updating neighborhood from neuron " + str(winner))
    
    # for each neuron weight
    for i in range(len(weights)):
        
        # get the distance between each neuron and the winner
        d = get_distance (i, winner)
        
        # for each weight of this neuron
        for j in range(len(weights[i])):
            
            # update the neuron weight
            weights[i][j] = weights[i][j] + neighbourhood_influence(d, iteration, epochs) * learning_rate(iteration, epochs) * (sample[j] - weights[i][j])
        
    return
    
def get_distance(n1, n2):
    
    # calculate the distance between this neuron and the BMU
    x_n1 = coordinate_map[n1][0]
    y_n1 = coordinate_map[n1][1]
     
    x_n2 = coordinate_map[n2][0]
    y_n2 = coordinate_map[n2][1]
     
    # calculate delta x
    if (x_n1 > x_n2):
        delta_x = x_n1 - x_n2
    else:
        delta_x = x_n2 - x_n1
         
    # calculate delta y
    if (y_n1 > y_n2):
        delta_y = y_n1 - y_n2
    else:
        delta_y = y_n2 - y_n1
     
    return (delta_x + delta_y)
    
def neighbourhood_influence (d, iteration, epochs):
    
    starting_neighbourhood_influence = 0.6
    
    sigma = (starting_neighbourhood_influence * exp(-iteration / epochs))
    
    return (exp( -d / (2 * (pow(sigma, 2)) ))) 
    
def learning_rate(iteration, epochs):
    
    return (initial_learning_rate * exp(-iteration / epochs))
    
def check_boundary (neuron, type = "available"):
    
    if neuron is None:
        return []
    
    x = coordinate_map[neuron][0]
    y = coordinate_map[neuron][1]
    
    all_boundary = ["L", "R", "B", "T"]
    boundary = []
    
    for i in range (len(coordinate_map)):
        
        # checking on left
        if (coordinate_map[i][0] == coordinate_map[neuron][0]-1 and coordinate_map[i][1] == coordinate_map[neuron][1]):
            boundary.append("L")
            
        # checking on right
        if (coordinate_map[i][0] == coordinate_map[neuron][0]+1 and coordinate_map[i][1] == coordinate_map[neuron][1]):
            boundary.append("R")
            
        # checking on bottom
        if (coordinate_map[i][0] == coordinate_map[neuron][0] and coordinate_map[i][1] == coordinate_map[neuron][1]-1):
            boundary.append("B")
            
        # checking on top
        if (coordinate_map[i][0] == coordinate_map[neuron][0] and coordinate_map[i][1] == coordinate_map[neuron][1]+1):
            boundary.append("T")

    if (type == "available"):
        return (list(set(all_boundary) - set(boundary)))
    elif (type == "unavailable"):
        return boundary
    
def grow (neuron, boundary):
    
    #print("growing on " + str(neuron) + " boundary: " + boundary)
    
    # append the weights
    add_weight(neuron, boundary)
    
    # append the coordinate map
    
    if (boundary == "L"): 
        coordinate_map.append([coordinate_map[neuron][0]-1, coordinate_map[neuron][1]])
    if (boundary == "R"): 
        coordinate_map.append([coordinate_map[neuron][0]+1, coordinate_map[neuron][1]])
    if (boundary == "B"): 
        coordinate_map.append([coordinate_map[neuron][0], coordinate_map[neuron][1]-1])
    if (boundary == "T"): 
        coordinate_map.append([coordinate_map[neuron][0], coordinate_map[neuron][1]+1])
    
    # append the acumulated error vector
    acumulated_error.append(0)
    
    return
    
def add_weight(neuron, boundary):
    
    x = coordinate_map[neuron][0]
    y = coordinate_map[neuron][1]
    
    # get the first neighbours
    first_neighbours = check_boundary(neuron, "unavailable")
    
    ##### ---------- Case A ---------- #####
    # There are two consecutive neighbours
    if oposite_side(boundary) in first_neighbours:
        
        if (oposite_side(boundary) == "L"):
            neuron2 = get_neuron_by_coordinate_map (x-1, y)
        
        if (oposite_side(boundary) == "R"):
            neuron2 = get_neuron_by_coordinate_map (x+1, y)
            
        if (oposite_side(boundary) == "B"):
            neuron2 = get_neuron_by_coordinate_map (x, y-1)
            
        if (oposite_side(boundary) == "T"):
            neuron2 = get_neuron_by_coordinate_map (x, y+1)
    
    ##### ---------- Case C ---------- #####
    # No consecutive neighbours, get the first one
    else:
        if (first_neighbours[0] == "L"):
            neuron2 = get_neuron_by_coordinate_map (x-1, y)
        
        if (first_neighbours[0] == "R"):
            neuron2 = get_neuron_by_coordinate_map (x+1, y)
            
        if (first_neighbours[0] == "B"):
            neuron2 = get_neuron_by_coordinate_map (x, y-1)
            
        if (first_neighbours[0] == "T"):
            neuron2 = get_neuron_by_coordinate_map (x, y+1)
            
    ##### --------------------------- #####
    ##### Calculating the new weights #####
    
    weights_width = len(weights[0])
    new_weights = []
    for i in range (weights_width):
        
        if (weights[neuron2][i] > weights[neuron][i]):
            new_weights.append(weights[neuron][i] - (weights[neuron2][i] - weights[neuron][i]))
        else:
            new_weights.append(weights[neuron][i] + (weights[neuron][i] - weights[neuron2][i]))
    
    weights.append(new_weights)
    
    return
    
def get_neuron_by_coordinate_map (x, y):
    
    for i in range(len(coordinate_map)):
        if (coordinate_map[i][0] == x and coordinate_map[i][1] == y):
            return i
    
    return None
    
def oposite_side (side):
    
    if side == "L":
        return "R"
    if side == "R":
        return "L"
    if side == "T":
        return "B"
    if side == "B":
        return "T"    
    
    return None
    
def spread_error (neuron):
    
    x = coordinate_map[neuron][0]
    y = coordinate_map[neuron][1]
    
    for i in range(len(coordinate_map)):
        
        x_temp = coordinate_map[i][0]
        y_temp = coordinate_map[i][1]
        
        # spreading to left
        if (x_temp == x-1 and y_temp == y):
            acumulated_error[i] = acumulated_error[i] + fd * acumulated_error[i]
        
        # spreading to right
        if (x_temp == x+1 and y_temp == y):
            acumulated_error[i] = acumulated_error[i] + fd * acumulated_error[i]
                    
        # spreading to bottom
        if (x_temp == x and y_temp == y-1):
            acumulated_error[i] = acumulated_error[i] + fd * acumulated_error[i]
                    
        # spreading to top
        if (x_temp == x and y_temp == y+1):
            acumulated_error[i] = acumulated_error[i] + fd * acumulated_error[i]       
    
    # decrease the error of the winner
    acumulated_error[neuron] = gt/2
    
    return
    
def print_clustering(input):
    
    print ("-----------------------")
    print ("Clusters")
    print ("-----------------------")

    for i in range (len(input)):
        
        bmu = find_best_matching_unit(input[i])
        print (bmu["winner"])
        
#    print ("-----------------------")
#    print ("Coordinate Map")
#    print ("-----------------------")
    
#    for i in range (len(coordinate_map)):
#        print (coordinate_map[i])
        
    # saving the neuron weights to a file
    with open('weights.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % weight for weight in weights)

    return
    
def plot_map (input, labels, neuron_labels, labels_colors, show_samples = False):

    x = []
    y = []
    
    # get the x, y coordinates
    for i in range(len(coordinate_map)):
        x.append(coordinate_map[i][0])
        y.append(coordinate_map[i][1])
        
    # get the labels
    classification_labels = []
    
    for i in range(len(weights)):
        classification_labels.append("N"+str(i))
        
    # get the colors for the neurons
    colors = []
    for i in range(len(weights)):
    
        if neuron_labels[i] == "":
            colors.append("#FFFFFF")
        else:
            colors.append(labels_colors[neuron_labels[i]])
    
    if (show_samples):
        for i in range(len(input)):
        
            bmu = find_best_matching_unit(input[i])
            n = bmu["winner"]
    
            classification_labels[n] += "\n" + str(labels[i])
        
        
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=500, c=colors, alpha=1.0)
        
    for i, txt in enumerate(classification_labels):
        ax.annotate(txt, (x[i], y[i]), size=7)
    
    ax.grid(True)
    plt.show()

    return
    
def get_neuron_labels (input, input_labels):
    
    neuron_labels = dict()
    frequency = dict()

    # extract the unique labels, and create a dict
    labels_set = set(input_labels)
    labels_unique = (list(labels_set))
    
    # create the frequency vector
    for i in range(len(weights)):
        frequency[i] = { i : 0 for i in labels_unique }

    # count the frequency of each label on each neuron
    for i in range(len(input)):
        
        bmu = find_best_matching_unit(input[i])
        n = bmu["winner"]
        
        frequency[n][input_labels[i]] += 1
    
    # set the most frequent label as the neuron label
    for i in range(len(weights)):
        
        count = 0
        top_label = ""
        
        for label in frequency[i]:
            
            if frequency[i][label] > count:
                count = frequency[i][label]
                top_label = label
         
        neuron_labels[i] = top_label
    
    return neuron_labels
    
def check_neuron_accuracy (input, neuron_labels, input_labels):
    
    neuron_stats = []
    
    # initialize neuron stats
    # [0] = hit
    # [1] = miss
    for i in range(len(weights)):
        neuron_stats.append([0, 0])
        
    # check each input classification, against neuron label
    for i in range(len(input)):
        
        bmu = find_best_matching_unit(input[i])
        n = bmu["winner"]
        
        if input_labels[i] == neuron_labels[n]:
            neuron_stats[n][0] += 1
        else:
            neuron_stats[n][1] += 1
    
    # printing neuron stats
    for i in range(len(neuron_stats)):
        
        sum = neuron_stats[i][0] + neuron_stats[i][1]

        if (sum == 0):
            print ("neuron [" + str(i) + "]: empty")
        else:
            print ("neuron [" + str(i) + "] [" + neuron_labels[i] + "]:" + str(100 * neuron_stats[i][0]/sum) + "%")
    
    return
    
def generate_confusion_matrix(input, neuron_labels, input_labels):
    
    frequency = dict()

    # extract the unique labels, and create a dict
    labels_set = set(input_labels)
    labels_unique = (list(labels_set))
    
    print ("------")

    # create the label hit/miss vector
    label_hit = { i : 0 for i in labels_unique }
    label_miss = { i : 0 for i in labels_unique }
    
    total_hit = 0
    total_miss = 0
    
    # check each classification to generate the confusion matrix
    for i in range(len(input)):
        
        bmu = find_best_matching_unit(input[i])
        n = bmu["winner"]
        
        if input_labels[i] == neuron_labels[n]:
            label_hit[input_labels[i]] += 1
            total_hit += 1
        else:
            label_miss[input_labels[i]] += 1
            total_miss += 1
    
    for l in labels_unique:
        print ("class [" + l + "] : " + str(100 * label_hit[l]/(label_hit[l] + label_miss[l])) + "%")
        
    print ("Total Accuracy : " + str(100 * total_hit/(total_hit + total_miss)) + "%")

    return
    