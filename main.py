from pennylane import numpy as np
import time
import training
import data
import plot
import testing

def main(initial_weights,epochs,pulsar_samples,non_pulsar_samples,test_pulsar_samples,test_non_pulsar_samples,quantum_circuit):
    global weights_crossEntropy,probability_non_pulsar,probability_pulsar,train_data,opt_prob_non_pulsar,classifications,probabilities,prob_pulsar_test,prob_non_pulsar_test
    '''Initializing constants'''
    square_loss_choice = 0
    cross_entropy_choice = 1
    '''Initializing constants'''
    #Finding the probability of finding pulsars and non_pulsars
    #Note, the pulsar_samples and non_pulsar_samples have their 8th element as the classification
    probability_non_pulsar = training.pulsar_probability(non_pulsar_samples,initial_weights,quantum_circuit)
    probability_pulsar = training.pulsar_probability(pulsar_samples,initial_weights,quantum_circuit)
    plot.probabilities(probability_pulsar,probability_non_pulsar)
      
    #----------------------------NOW LETS TRAIN DATA----------------------------
    
    train_data = np.vstack((pulsar_samples, non_pulsar_samples))
    
    start_time = time.perf_counter()
    weights_crossEntropy,loss_crossEntropy = training.training(epochs, initial_weights, train_data, cross_entropy_choice,quantum_circuit)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time)
    
    opt_prob_non_pulsar = training.pulsar_probability(non_pulsar_samples,weights_crossEntropy,quantum_circuit)
    opt_prob_pulsar = training.pulsar_probability(pulsar_samples,weights_crossEntropy,quantum_circuit)
    plot.optimized_probabilities(opt_prob_non_pulsar,opt_prob_pulsar,epochs,cross_entropy_choice)
    plot.loss_function(epochs,loss_crossEntropy,cross_entropy_choice)
    
    #----------------------------NOW LETS TEST DATA----------------------------
    
    #test_features,test_class = data.feature_class_split(test_set)
    prob_pulsar_test = training.pulsar_probability(test_pulsar_samples,weights_crossEntropy,quantum_circuit)
    prob_non_pulsar_test = training.pulsar_probability(test_non_pulsar_samples,weights_crossEntropy,quantum_circuit)
    probabilities = np.concatenate((prob_pulsar_test, prob_non_pulsar_test))
    classifications = np.vstack((test_pulsar_samples,test_non_pulsar_samples))[:, 8]
    
    sensitivity,specificity,threshold = testing.calculate_sensitivity_specificity(probabilities, classifications)
    print("Sensitivity = {}%".format(sensitivity*100))
    print("Specificity = {}%".format(specificity*100))
    plot.test_probabilities(prob_pulsar_test,prob_non_pulsar_test,epochs,cross_entropy_choice,threshold,sensitivity,specificity)




initial_weights = 2 * np.pi * np.random.random(size=(9, 3))#, requires_grad=True) 
epochs = 10

normalized_dataset = data.normalize()
    
pulsar_samples,non_pulsar_samples,test_pulsar_samples,test_non_pulsar_samples = data.sample_pulsars(normalized_dataset,train_size=100,test_size=1000)

import old_quantum_circuit as quantum_circuit
main(initial_weights,epochs,pulsar_samples,non_pulsar_samples,test_pulsar_samples,test_non_pulsar_samples,quantum_circuit)






