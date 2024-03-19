import numpy as np
from scipy import ndimage
from visualization.view_2D import plot_parallel
import matplotlib.pyplot as plt


class vesselness2d:
    def __init__(self, image, sigma, tau):
        super(vesselness2d, self).__init__()

        self.image = image
        self.sigma = sigma
        self.tau = tau
        self.size = image.shape

    def gaussian_filter(self, image, sigma):
        image = ndimage.gaussian_filter(image, sigma, mode='nearest')
        return image

    def gradient_2d(self, np_array, option):
        x_size = self.size[0]
        y_size = self.size[1]
        gradient = np.zeros(np_array.shape)
        if option == "x":
            gradient[0, :] = np_array[1, :] - np_array[0, :]
            gradient[x_size - 1, :] = np_array[x_size - 1, :] - np_array[x_size - 2, :]
            gradient[1:x_size - 2, :] = \
                (np_array[2:x_size - 1, :] - np_array[0:x_size - 3, :]) / 2
        else:
            gradient[:, 0] = np_array[:, 1] - np_array[:, 0]
            gradient[:, y_size - 1] = np_array[:, y_size - 1] - np_array[:, y_size - 2]
            gradient[:, 1:y_size - 2] = \
                (np_array[:, 2:y_size - 1] - np_array[:, 0:y_size - 3]) / 2
        return gradient

    def Hessian2d(self, image, sigma):
        # print(sigma)
        image = ndimage.gaussian_filter(image, sigma, mode='nearest')
        Dy = self.gradient_2d(image, "y")
        Dyy = self.gradient_2d(Dy, "y")

        Dx = self.gradient_2d(image, "x")
        Dxx = self.gradient_2d(Dx, "x")
        Dxy = self.gradient_2d(Dx, 'y')
        return Dxx, Dyy, Dxy

    def eigval_Hessian2d(self, Dxx, Dyy, Dxy):
        tmp = np.sqrt((Dxx - Dyy) ** 2 + 4 * (Dxy ** 2))
        # compute eigenvectors of J, v1 and v2
        mu1 = 0.5 * (Dxx + Dyy + tmp)
        mu2 = 0.5 * (Dxx + Dyy - tmp)
        # Sort eigen values by absolute value abs(Lambda1) < abs(Lambda2)
        indices = (np.absolute(mu1) > np.absolute(mu2))
        Lambda1 = mu1
        Lambda1[indices] = mu2[indices]

        Lambda2 = mu2
        Lambda2[indices] = mu1[indices]
        return Lambda1, Lambda2

    def imageEigenvalues(self, I, sigma):
        hxx, hyy, hxy = self.Hessian2d(I, sigma)
        # hxx, hyy, hxy = self.Hessian2d(I, sigma)
        c = sigma ** 2
        hxx = -c * hxx
        hyy = -c * hyy
        hxy = -c * hxy

        B1 = -(hxx + hyy)
        B2 = hxx * hyy - hxy ** 2
        T = np.ones(B1.shape)
        T[(B1 < 0)] = 0
        T[(B1 == 0) & (B2 == 0)] = 0
        T = T.flatten()
        indeces = np.where(T == 1)[0]
        hxx = hxx.flatten()
        hyy = hyy.flatten()
        hxy = hxy.flatten()
        hxx = hxx[indeces]
        hyy = hyy[indeces]
        hxy = hxy[indeces]
        #     lambda1i, lambda2i = hessian_matrix_eigvals([hxx, hyy, hxy])
        lambda1i, lambda2i = self.eigval_Hessian2d(hxx, hyy, hxy)
        lambda1 = np.zeros(self.size[0] * self.size[1], )
        lambda2 = np.zeros(self.size[0] * self.size[1], )

        lambda1[indeces] = lambda1i
        lambda2[indeces] = lambda2i

        # removing noise
        lambda1[(np.isinf(lambda1))] = 0
        lambda2[(np.isinf(lambda2))] = 0

        lambda1[(np.absolute(lambda1) < 1e-4)] = 0
        lambda1 = lambda1.reshape(self.size)

        lambda2[(np.absolute(lambda2) < 1e-4)] = 0
        lambda2 = lambda2.reshape(self.size)
        return lambda1, lambda2

    def vessel2d(self):
        for j in range(len(self.sigma)):
            lambda1, lambda2 = self.imageEigenvalues(self.image, self.sigma[j])
            lambda3 = lambda2.copy()
            new_tau = self.tau * np.min(lambda3)
            lambda3[(lambda3 < 0) & (lambda3 >= new_tau)] = new_tau
            different = lambda3 - lambda2
            response = ((np.absolute(lambda2) ** 2) * np.absolute(different)) * 27 / (
                    (2 * np.absolute(lambda2) + np.absolute(different)) ** 3)
            response[(lambda2 < lambda3 / 2)] = 1
            response[(lambda2 >= 0)] = 0

            response[np.where(np.isinf(response))[0]] = 0
            if j == 0:
                vesselness = response
            else:
                vesselness = np.maximum(vesselness, response)
        vesselness[(vesselness < 1e-2)] = 0
        return vesselness


def perform_filter(np_array, tau=1, enhance=False):
    new_array = np_array.copy()
    if enhance:
        new_array = np.sqrt(1 - (1 - new_array) ** 2)
    new_array = 255 * (1 - new_array)

    sigma = [0.5, 1, 1.5]
    output = vesselness2d(new_array, sigma, tau)
    output = output.vessel2d()

    return output


if __name__=='__main__':
    image = np.load("C:\\Users\\11231\\Desktop\\Desktop_Files\\shared_file\\TM-1.npz")["arr_0"]
    image = np.clip((image + 1000) / 1200, 0, 1)
    image = 255 - image[:, :, 205] * 255


    sigma = [0.5, 1, 1.5, 2, 2.5]
    tau = 2

    output = vesselness2d(image, sigma, tau)
    output, lambda1, lambda2, lambda_tau, gamma = output.vessel2d(kappa1=0.5, kappa2=0.5)
    print(np.max(output), np.min(output))
    plt.imshow(output, cmap="gray")
    plt.show()

    plot_parallel(
        # a=image,
        b=output,
        c1=lambda1,
        c2=lambda2,
        c3=lambda_tau,
        c4=gamma
    )

    # plt.figure(figsize=(10, 10))
    # plt.imshow(output, cmap='gray')
