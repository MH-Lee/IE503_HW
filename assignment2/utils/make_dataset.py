import numpy as np
import matplotlib.pyplot as plt

np.random.seed(110)


def make_cov(mode='spherical', size=(2,2)):
    if mode == 'spherical':
        multiplier = np.random.randint(3, 10)
        cov = np.eye(size[0])*multiplier
    else:
        i = 0
        min_eig = -1
        while min_eig < 0:
            sign = np.random.choice([1,-1])
            cov = np.random.randint(3, 10 ,size=size)
            cov[0,1] = cov[0,1] * sign
            cov[1,0] = cov[0,1]
            min_eig = np.min(np.real(np.linalg.eigvals(cov)))
            i+=1
        print(i)
    return cov

def make_dataset(sample_n =1000, mode="balanced", variance='spherical'):
    c1_rate = 0.5 if mode == 'balanced' else round(np.random.uniform(0.8,0.9,1)[0], 2)
    c2_rate = round(1 - c1_rate,2)
    mu1 = np.random.normal(3,1,size=(2,)).round(2)
    mu2 = np.random.normal(10,2,size=(2,)).round(2)
    # rand_n1, rand_n1_1 = np.random.choice(np.linspace(3,6,7)), np.random.choice([n for n in np.linspace(-3,3,9) if n != 0])
    # cov1 = np.eye(2) * rand_n1 if variance == 'spherical' else np.array([[np.random.randint(4, 7), rand_n1_1],[rand_n1_1, np.random.randint(4, 7)]])
    # rand_n2, rand_n2_1 = np.random.choice(np.linspace(2,5,7)), np.random.choice([n for n in np.linspace(-2,2,6) if n != 0])
    # cov2 = np.eye(2) * rand_n2 if variance == 'spherical' else np.array([[np.random.randint(2, 4), rand_n2_1],[rand_n2_1, np.random.randint(3, 5)]])
    cov1 = make_cov(mode=variance)
    cov2 = make_cov(mode=variance)
    print("sample rate {} : {}".format(c1_rate, c2_rate))
    print("mu1 :", mu1, "cov1 :", cov1)
    print("mu2 :", mu2, "cov2 :", cov2)
    X1, class1 = np.random.multivariate_normal(mu1, cov1, int(sample_n*c1_rate)), np.array([1]*int(sample_n*c1_rate)).reshape(-1,1)
    X2, class2 = np.random.multivariate_normal(mu2, cov2, int(sample_n*c2_rate)), np.array([2]*int(sample_n*c2_rate)).reshape(-1,1)
    data = np.append(X1, X2,axis=0)
    y_label = np.append(class1, class1,axis=0)
    print(X1.shape, class1.shape)
    print(X2.shape, class2.shape)
    plt.scatter(X1[:,0], X1[:,1], marker='o', s=40, color='tab:blue', label='class1')
    plt.scatter(X2[:,0], X2[:,1], marker='^', s=40, color='tab:red', label='class2')
    plt.legend()
    plt.axis('equal')
    plt.show()
    return data, y_label


# data, y_label = make_dataset(mode='imbalanced', variance='non-spherical')
