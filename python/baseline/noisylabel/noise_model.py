import numpy as np
from utilities import minDiagonalSwap


class inject_noise(object):
    # class fo injecting artificial noise to the clean labels.

    def __init__(self, y_train, n_type, noiselvl, noise_mat=np.zeros(0)):
        """
        type(n_type): str: noise type ['uni','rand','CC']
        type(y_train): Numpy array: clean training labels
        type(noiselvl): float: input levelof noise
        """
        self.n_cls = len(list(set(y_train)))
        self.type = n_type
        self.noiselvl = noiselvl
        self.true_label = y_train
        if not noise_mat.any():
            self.noise_matrix = self.genNoiseMat()
        else:
            self.noise_matrix = noise_mat
        # print(self.noise_matrix)
        self.noisy_label = self.generateLabels()
        print('object created')

    def genNoiseMat(self):
        """Generate the random/uniform noise matrix
        return: a stochastic matrix based on the type of noise.
        """
        n_class = self.n_cls
        noise_matrix = np.zeros([n_class, n_class], dtype='float')

        if self.n_cls <= 2:
            noise_matrix = (1-self.noiselvl)*np.eye(n_class, dtype='float') + self.noiselvl*(np.ones(n_class, dtype='float') - np.eye(n_class, dtype='float'))
        else:
            # Defines random noise over unit simplex
            if self.type == 'rand':
                for a in range(n_class):
                    nums = [np.random.randint(0, 10)*1.0 for x in  range(n_class)]
                    # print(nums[1]/sum(np.array(nums)))
                    nums = self.noiselvl*(nums/sum(np.array(nums)))
                    # print(nums)
                    minDiagonalSwap(nums, a)
                    noise_matrix[a, :] = nums
                    # print(noise_matrix)
                noise_matrix = (1-self.noiselvl)*np.eye(n_class, dtype='float') + noise_matrix

            # defines uniform noise
            elif self.type == 'uni':
                noise_matrix = ((1-self.noiselvl)*np.eye(n_class, dtype='float') + (self.noiselvl/n_class)*(np.ones(n_class, dtype='float')))
            # TO-DO
            # define other type of noise
            elif self.type == 'cc':
                levels = [np.random.randint(0,n_class)*1.0 for i in range(n_class)]
                levels = n_class*self.noiselvl*(levels/sum(np.array(levels)))
                ind = 0
                for n in levels:
                    nums = [np.random.randint(0, 10)*1.0 for x in  range(n_class)]
                    # print(nums[1]/sum(np.array(nums)))
                    nums = n*(nums/sum(np.array(nums)))
                    # print(nums)
                    minDiagonalSwap(nums, ind)
                    noise_matrix[ind, :] =(1-n)*np.eye(n_class, dtype='float')[ind] + nums
                    ind = ind + 1
                print(noise_matrix)
            else:
                print("Please define correct form of noise ['uni','rand','cc']")
                return None
        return noise_matrix

    def generateLabels(self):
        """Generate noisy labels for the training and valid set
        return: Array of noisy labels.
        """

        noisy_label = np.zeros_like(self.true_label)
        classes = list(set(self.true_label))

        for cls in range(self.n_cls):
            flip_label = []
            n_cls_labels = sum(self.true_label == classes[cls])

            for _ in range(n_cls_labels):
                flip_label.append(np.random.choice(np.arange(0, self.n_cls), p=self.noise_matrix[cls,:]))

            flip_label = np.array(flip_label)

            noisy_label[self.true_label == cls] = flip_label.astype(int)
            print('Number of labels for class ', cls, ' flipped: ', n_cls_labels - sum(flip_label == cls), 'out of ', n_cls_labels)

        return noisy_label


if __name__ == '__main__':
    y_train = np.array([0, 0, 1, 1, 0, 1, 0, 0, 0, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 0, 2, 4, 4, 2, 4, 2, 4, 2])
    print(y_train)
    noise_val = inject_noise(y_train, n_type='rand', noiselvl=0.1)
    print(noise_val.type)
    print(noise_val.noiselvl)
    print(noise_val.n_cls)
    print(noise_val.noise_matrix)
    print(type(noise_val.noisy_label[0]))