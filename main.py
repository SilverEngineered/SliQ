import triplet_generator
import sliq
import basic_net
import numpy as np
if __name__ == '__main__':
    dataset='MNIST'
    num_qubits = 11
    # Preprocessing
    triplets, indices, labels = triplet_generator.generate_pca_triplets(dataset, label_space=2, num_triplets=1000, testing=True)
    network = sliq.Triplet(num_qubits)
    network.train(triplets)
