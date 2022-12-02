import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import AmplitudeEmbedding
import numpy
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import random
from PIL import Image
from os import listdir
from os.path import isfile, join
from triplet_generator import get_color_density
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import itertools
import pandas as pd
import os
class Triplet:
    def __init__(self, num_qubits):
        self.weights_list = []
        self.num_wires = num_qubits
        self.num_layers = 4
        self.batch_size = 30
        self.epochs = 501
        self.embed_dims = 5
        self.losses = []
    def train(self, triplets):
        opt = qml.GradientDescentOptimizer(stepsize=0.1)
        self.weights = 0.01 * np.random.randn(self.num_layers, self.num_wires, 3)
        for i in range(self.epochs):
            self.cur_epoch = i
            batch_index = np.random.randint(0, len(triplets), (self.batch_size,))
            x_train_batch = [triplets[im] for im in batch_index]
            self.weights = opt.step(lambda v: self.cost_embedding(v, x_train_batch), self.weights)
            self.losses.append(self.cost_embedding(self.weights, x_train_batch))
            if i %20 == 0:
                print(np.mean(self.losses))
                self.losses = []
                self.weights_list.append(self.weights)
                if i > 1:
                    numpy.save('base_land', self.weights_list)

    def cost_embedding(self, weights, features):
        loss = 0
        for im in features:
            # Correct Order
            flattened = []
            for i, j, k in zip(im[0], im[1], im[2]):
                flattened.append(i)
            A = self.embed(self, weights, np.array(flattened))
            flattened = []
            for i, j, k in zip(im[0], im[1], im[2]):
                flattened.append(j)
            P = self.embed(self, weights, np.array(flattened))
            flattened = []
            for i, j, k in zip(im[0], im[1], im[2]):
                flattened.append(k)
            N = self.embed(self, weights, np.array(flattened))
            loss += (np.square(A[0]-P[0]) + np.square(A[1]-P[1])) - (np.square(A[0]-N[0]) + np.square(A[1]-N[1]))
        #print(str(loss))
        #print(f'Epoch: {self.cur_epoch} Accuracy: {100* correct / (len(features))}')
        return loss / len(features)

    @qml.qnode(qml.device(name='default.qubit', wires=13))
    def embed(self, weights, features=None):
        AmplitudeEmbedding(features=features.astype('float64'), wires=range(self.num_wires), normalize=True, pad_with=0)
        #AmplitudeEmbedding(features=features.astype('float64'), wires=range(2), normalize=True, pad_with=0)
        for W in weights:
            self.layer(W)
        return [qml.expval(qml.PauliZ(i)) for i in range(2)]
    


    def layer(self, W):
        for i in range(self.num_wires):
            qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)
        for wire in range(self.num_wires-1):
            qml.CNOT(wires=[wire, self.num_wires-1])


    def eval_acc(self, triplets, labels, weights):
        all_weights = numpy.load(weights)
        for count, weights in enumerate(all_weights[-5:]):
            correct = 0
            embeddings = []
            for im in tqdm(triplets):
                flattened = []
                for i, j, k in zip(im[0], im[1], im[2]):
                    flattened.append(i)
                z_out1 = self.embed(self, weights, np.array(flattened))
                embeddings.append(z_out1)
            
            num_clusters = 3
            kmeans = KMeans(num_clusters)
            #kmeans = GaussianMixture(num_clusters)
            kmeans.fit(embeddings)
            #kmeans = GaussianMixture(2)
            y_hat = kmeans.predict(embeddings)
            y = [i[0] for i in labels]


            max_cor = 0
            for lab_set in [['CI', 'CA', 'CM'], ['CI', 'CM', 'CA'], ['CA', 'CM', 'CI'], ['CA', 'CI', 'CM'], ['CM', 'CA', 'CI'], ['CM', 'CI', 'CA']]:
                num_cor = 0
                lab_dict = {lab_set[0]: 0, lab_set[1]: 1, lab_set[2]: 2}
                for i, j in zip(y_hat, y):
                    if i == lab_dict[j]:
                        num_cor +=1
                max_cor = max(num_cor, max_cor)
            print(100*max_cor / len(triplets))
            y_hats = self.generate_y_hats(y_hat, num_clusters)
            for yh in y_hats:
                num_cor = 0
                for i, j in zip(yh, y):
                    if i == j:
                        num_cor +=1
                max_cor = max(num_cor, max_cor)
            print(100*max_cor / len(triplets))
    def generate_y_hats(self, y_hat, num_clusters):
        import itertools
        perms = list(itertools.permutations(range(num_clusters)))
        y_hats = []
        for perm in perms:
            this_yhat = [perm[i] for i in y_hat]
            y_hats.append(this_yhat)
        return y_hats

    def eval_real_machine(self, csv_file):
        with open(os.path.join(csv_file)) as f:
            lines = f.read()
        lines = [tuple(i.replace(' ', '').split(',')) for i in lines.split('\n') if i!='']
        labels = [i[1] for i in lines]
        data = [i[2:] for i in lines]
        float_data = []
        for t in data:
            float_data.append([float(o) for o in t])
        data = float_data
        kmeans = GaussianMixture(3)
        kmeans.fit(data)
        y_hat = kmeans.predict(data)
        y = [i[0] for i in labels]

        max_cor = 0
        for lab_set in [['CI', 'CA', 'CM'], ['CI', 'CM', 'CA'], ['CA', 'CM', 'CI'], ['CA', 'CI', 'CM'], ['CM', 'CA', 'CI'], ['CM', 'CI', 'CA']]:
            num_cor = 0
            lab_dict = {0: lab_set[0], 1: lab_set[1], 2: lab_set[2]}
            for i, j in zip(y_hat, labels):
                if lab_dict[i] == j:
                    num_cor +=1
            max_cor = max(num_cor, max_cor)
        print(100*max_cor / len(data))

    def find_closest(self, triplets, indecies):
        #weights = 0.01 * np.random.randn(self.num_layers, self.num_wires, 3)
        weights = numpy.load('base_land.npy')[-1]
        path = 'datasets/Landscapes'
        files = [f for f in listdir(path) if isfile(join(path, f))]
        files.sort()
        all_losses = []
        all_distances = []
        for l in tqdm(range(len(triplets))):
            distance_tuples = []
            loss_tuples = []
            min_loss = 10e5
            anchor_image = triplets[l][0]
            anchor_index = indecies[l][0]
            dists = []
            closest_dist = 10e5
            for image, indexes in zip(triplets, indecies):
                if indexes[0] != anchor_index:
                    flattened = []
                    for i, j, in zip(anchor_image, image[0]):
                        flattened.append(i)
                    #flattened.append(j)
                    loc1 = self.embed(self, weights, features=np.array(flattened))
                    flattened = []
                    for i, j, in zip(anchor_image, image[0]):
                        flattened.append(j)
                    #flattened.append(i)
                    #loss = np.abs(loss1 - loss2)
                    loc2 = self.embed(self, weights, features=np.array(flattened))
                    loss = np.abs(loc1[0] - loc2[0]) + np.abs(loc1[1] - loc2[1])

                    im = Image.open('datasets/Landscapes/' + str(files[anchor_index]))
                    new_width, new_height = 80, 80
                    width, height = im.size
                    left = (width - new_width)//2
                    top = (height - new_height)//2
                    right = (width + new_width)//2
                    bottom = (height + new_height)//2
                    im1= im.crop((left, top, right, bottom))


                    im = Image.open('datasets/Landscapes/' + str(files[indexes[0]]))
                    new_width, new_height = 80, 80
                    width, height = im.size
                    left = (width - new_width)//2
                    top = (height - new_height)//2
                    right = (width + new_width)//2
                    bottom = (height + new_height)//2
                    im2= im.crop((left, top, right, bottom))

                    #print(loss1, loss2, anchor_index, indexes[0])
                    anchor_r, anchor_g, anchor_b = get_color_density(np.array(im1))
                    new_r, new_g, new_b = get_color_density(np.array(im2))
                    dist = np.sum(np.power(np.array(anchor_r+anchor_g+anchor_b)- np.array(new_r+new_g+new_b),2)) ** .5
                    dists.append(dist)
                    distance_tuples.append((float(dist), files[indexes[0]]))
                    loss_tuples.append((float(loss), files[indexes[0]]))
                    if loss < min_loss:
                        min_loss = loss
                        closest_index = indexes[0]
                        closest_dist = dist
            im = Image.open('datasets/Landscapes/' + str(files[anchor_index]))
            new_width, new_height = 80, 80
            width, height = im.size
            left = (width - new_width)//2
            top = (height - new_height)//2
            right = (width + new_width)//2
            bottom = (height + new_height)//2
            im1= im.crop((left, top, right, bottom))

            im2 = Image.open('datasets/Landscapes/' + str(files[closest_index]))
            new_width, new_height = 80, 80
            width, height = im2.size
            left = (width - new_width)//2
            top = (height - new_height)//2
            right = (width + new_width)//2
            bottom = (height + new_height)//2
            im2 = im2.crop((left, top, right, bottom))
            Image.fromarray(np.hstack((np.array(im1),np.array(im2)))).save(f'BasicImages/{l}.jpg')
            distance_tuples.sort()
            loss_tuples.sort()
            
            all_losses.append(loss_tuples)
            all_distances.append(distance_tuples)
            numpy.save('rankings_big_base', {'all_distances': all_distances, 'all_losses': all_losses})
            