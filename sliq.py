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
        self.batch_size = 20
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
                #numpy.save('aids_new', self.weights_list)


    def cost_embedding(self, weights, features):
        loss = 0
        for im in features:
            # Correct Order
            sign = 1
            flattened = []


            # #good
            for i, j, k in zip(im[0], im[1], im[2]):
                flattened.append(i)
                flattened.append(j)


            #bad ends also cycle loss
            z_out1 = self.embed(self, weights, np.array(flattened))
            z1_loss = self.triplet_loss(z_out1)
            # Reversed Order
            sign = -1
            flattened = []
            for i, j, k in zip(im[0], im[1], im[2]):
                flattened.append(k)
                flattened.append(i)
            z_out2 = self.embed(self, weights, np.array(flattened))
            z2_loss = self.triplet_loss(z_out2)
            siam_loss = .9*(z1_loss - z2_loss)
            consistancy_loss = .1*(np.abs((z_out1[0] - z_out2[2])) + np.abs((z_out1[1] - z_out2[3])))
            loss += .9*siam_loss
            loss += .1*consistancy_loss
        return loss / len(features)

    @qml.qnode(qml.device(name='default.qubit', wires=11))
    def embed(self, weights, features=None):
        AmplitudeEmbedding(features=features.astype('float64'), wires=range(self.num_wires), normalize=True, pad_with=0)
        #AmplitudeEmbedding(features=features.astype('float64'), wires=range(3), normalize=True, pad_with=0)
        for W in weights:
            self.layer(W)
        return [qml.expval(qml.PauliZ(i)) for i in range(4)]
    


    def layer(self, W):
        for i in range(self.num_wires):
            qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)
        for wire in range(self.num_wires-1):
            qml.CNOT(wires=[wire, self.num_wires-1])
    def triplet_loss(self, z_out):
        return np.abs(z_out[0] - z_out[2]) + np.abs(z_out[1] - z_out[3])
    def find_closest(self, triplets, indecies):
        weights = numpy.load('landscape_weights_4_layers.npy')
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
                        flattened.append(j)
                    loc1 = self.embed(self, weights, features=np.array(flattened))
                    flattened = []
                    for i, j, in zip(anchor_image, image[0]):
                        flattened.append(j)
                        flattened.append(i)
                    loc2 = self.embed(self, weights, features=np.array(flattened))
                    loss = sum([np.abs(i-j) for i,j in zip(loc1, loc2)])




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
            distance_tuples.sort()
            loss_tuples.sort()
            
            all_losses.append(loss_tuples)
            all_distances.append(distance_tuples)
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
            Image.fromarray(np.hstack((np.array(im1),np.array(im2)))).save(f'SliqImages/{l}.jpg')


    def eval_acc(self, triplets, labels, weights_file):
        x = [i[0] for i in triplets]
        y = [i[0] for i in labels]
        all_weights = numpy.load(weights_file)
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        x = [i[1] for i in triplets]
        y = [i[1] for i in labels]
        clf.fit(x, y)
        print(clf.score(x, y))
        accuracies = []
        print(len(all_weights))
        for count, weights in enumerate(all_weights[-5:]):
            correct = 0
            embeddings = []
            for im in tqdm(triplets):
                flattened = []
                for i, j, k in zip(im[0], im[1], im[2]):
                    flattened.append(i)
                    flattened.append(j)
                z_out1 = self.embed(self, weights, np.array(flattened))
                embeddings.append(np.reshape(z_out1, [-1]))

            if 'mnist' not in weights_file and 'fashion' not in weights_file:
                kmeans = GaussianMixture(3)
                kmeans.fit(embeddings)
                y_hat = kmeans.predict(embeddings)
                y = [i[0] for i in labels]
                color_map={'CI': 'r', 'CM': 'b', 'CA': 'g'}
                for count, l in enumerate(embeddings):
                    plt.scatter(l[0], l[1], color=color_map[y[count]])
                plt.show()
                max_cor = 0
                for lab_set in [['CI', 'CA', 'CM'], ['CI', 'CM', 'CA'], ['CA', 'CM', 'CI'], ['CA', 'CI', 'CM'], ['CM', 'CA', 'CI'], ['CM', 'CI', 'CA']]:
                    num_cor = 0
                    lab_dict = {lab_set[0]: 0, lab_set[1]: 1, lab_set[2]: 2}
                    for i, j in zip(y_hat, y):
                        if i == lab_dict[j]:
                            num_cor +=1
                    max_cor = max(num_cor, max_cor)
                print(100*max_cor / len(triplets))
                
                accuracies.append(100*max_cor / len(triplets))
            else:
                num_clusters = 2
                if '4_classes' in weights_file:
                    num_clusters = 4
                kmeans = GaussianMixture(num_clusters)
                kmeans.fit(embeddings)
                y_hat = kmeans.predict(embeddings)
                y = [i[0] for i in labels]
                max_cor = 0
                y_hats = self.generate_y_hats(y_hat, num_clusters)
                for yh in y_hats:
                    num_cor = 0
                    for i, j in zip(yh, y):
                        if i == j:
                            num_cor +=1
                    max_cor = max(num_cor, max_cor)
                print(100*max_cor / len(triplets))
                accuracies.append(100*max_cor / len(triplets))
        print(accuracies)

    def generate_y_hats(self, y_hat, num_clusters):
        import itertools
        perms = list(itertools.permutations(range(num_clusters)))
        y_hats = []
        for perm in perms:
            this_yhat = [perm[i] for i in y_hat]
            y_hats.append(this_yhat)
        return y_hats
        
    def eval_consistancy(self, triplets, weights):
        embeddings = []
        weights_name=weights
        weights = np.load(weights)[-1]
        reversed_embeddings = []
        for im in tqdm(triplets):
            flattened = []
            for i, j, k in zip(im[0], im[1], im[2]):
                flattened.append(i)
                flattened.append(j)
            z_out = self.embed(self, weights, np.array(flattened))
            embeddings.append(z_out)
            flattened = []
            for i, j, k in zip(im[0], im[1], im[2]):
                flattened.append(j)
                flattened.append(i)
            z_out = self.embed(self, weights, np.array(flattened))
            reversed_embeddings.append(z_out)
        losses = []
        for i, j in zip(embeddings, reversed_embeddings):
            #print(i,j)
            #losses.append(float(np.abs(i[0] - j[2]) + float(np.abs(i[1] - j[3]))))
            losses.append(float(np.abs(i[0] - j[2]) + float(np.abs(i[1] - j[3])) + float(np.abs(i[2] - j[0])) + float(np.abs(i[3] - j[1]))))
        print(np.mean(losses))
        return losses
        
    def eval_real_machine(self, csv_file):
        with open(os.path.join(csv_file)) as f:
            lines = f.read()
        lines = [tuple(i.replace(' ', '').split(',')) for i in lines.split('\n') if i!='']
        r_ind = np.random.randint(0, len(lines), (395,))
        labels = [i[1] for i in lines]
        data = [i[5:] for i in lines]
        labels = list(np.array(labels)[r_ind])
        data = list(np.array(data)[r_ind])
        float_data = []
        for t in data:
            float_data.append([float(o) for o in t])
        data = float_data
        kmeans = GaussianMixture(3)
        kmeans = KMeans(3)
        kmeans.fit(data)
        y_hat = kmeans.predict(data)
        y = [i for i in labels]

        max_cor = 0
        for lab_set in [['CI', 'CA', 'CM'], ['CI', 'CM', 'CA'], ['CA', 'CM', 'CI'], ['CA', 'CI', 'CM'], ['CM', 'CA', 'CI'], ['CM', 'CI', 'CA']]:
            num_cor = 0
            lab_dict = {0: lab_set[0], 1: lab_set[1], 2: lab_set[2]}
            for i, j in zip(y_hat, labels):
                if lab_dict[i] == j:
                    num_cor +=1
            max_cor = max(num_cor, max_cor)
        print(100*max_cor / len(data))
    # def f_loss(self, lis):
    #     loss=0
    #     if len(lis)%2!=0:
    #         return None
    #     for i in range(int(len(lis)/2)):
    #         #print(i,i+len(lis)/2)
    #         loss = loss + np.abs(lis[i]-lis[int(i+len(lis)/2)])
    #     return loss