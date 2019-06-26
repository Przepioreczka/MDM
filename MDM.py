import numpy as np
import scipy
from scipy.linalg import eigvalsh
from numpy.core.numerictypes import typecodes

# Functions borrowed from alexandrebarachant/pyRiemann github repository
# https://github.com/alexandrebarachant/pyRiemann


def matrix_operator(Ci, operator):
    """matrix equivalent of an operator."""
    if Ci.dtype.char in typecodes["AllFloat"] and not np.isfinite(Ci).all():
        raise ValueError(
            "Covariance matrices must be positive definite. Add regularization to avoid this error."
        )
    eigvals, eigvects = scipy.linalg.eigh(Ci, check_finite=False)
    eigvals = np.diag(operator(eigvals))
    Out = np.dot(np.dot(eigvects, eigvals), eigvects.T)
    return Out


def logm(Ci):
    return matrix_operator(Ci, np.log)


def expm(Ci):
    return matrix_operator(Ci, np.exp)


def sqrtm(Ci):
    return matrix_operator(Ci, np.sqrt)


def invsqrtm(Ci):
    isqrt = lambda x: 1.0 / np.sqrt(x)
    return matrix_operator(Ci, isqrt)


def get_sample_weight(sample_weight, data):
    if sample_weight is None:
        sample_weight = np.ones(data.shape[0])
    if len(sample_weight) != data.shape[0]:
        raise ValueError("len of sample_weight must be equal to len of data.")
    sample_weight /= np.sum(sample_weight)
    return sample_weight


def mean_riemann(covmats, tol=10e-9, maxiter=50, init=None, sample_weight=None):
    sample_weight = get_sample_weight(sample_weight, covmats)
    Nt, Ne, Ne = covmats.shape
    if init is None:
        C = np.mean(covmats, axis=0)
    else:
        C = init
    k = 0
    nu = 1.0
    tau = np.finfo(np.float64).max
    crit = np.finfo(np.float64).max
    # stop when J<10^-9 or max iteration = 50
    while (crit > tol) and (k < maxiter) and (nu > tol):
        k = k + 1
        C12 = sqrtm(C)
        Cm12 = invsqrtm(C)
        J = np.zeros((Ne, Ne))

        for index in range(Nt):
            tmp = np.dot(np.dot(Cm12, covmats[index, :, :]), Cm12)
            J += sample_weight[index] * logm(tmp)

        crit = np.linalg.norm(J, ord="fro")
        h = nu * crit
        C = np.dot(np.dot(C12, expm(nu * J)), C12)
        if h < tau:
            nu = 0.95 * nu
            tau = h
        else:
            nu = 0.5 * nu

    return C


def distance_riemann(A, B):
    return np.sqrt((np.log(eigvalsh(A, B)) ** 2).sum())


def dist_riem(A, B):
    return np.linalg.norm(scipy.linalg.logm(np.dot(np.linalg.inv(A), B)))


# Proper code


class Base(object):
    #   Accepts 4 dimensional array:
    #   (x,y,z,s) - let it be the dimension
    #       x - how many gestures are to be classified
    #       y - number of repetitions for each gesture (equal for every gesture)
    #       z - number of channels
    #       s - number of samples in each gesture
    #   Returns 3 dimensional array with gestures classified based on riemann geometry
    #   which is necessary for classifying gestures with online signal
    def __init__(self, arr):
        self.__gestures = arr
        self.__number_of_gestures = arr.shape[0]
        self.__repetitions = arr.shape[1]
        self.__ch = arr.shape[2]
        self.__SPD_matrices = np.zeros(
            (self.__number_of_gestures, self.__ch, self.__ch)
        )

    #   matrix with matrices classified (one gesture - one 2 dimensional matrix)
    def Make_SPDBase(self):
        corr_package = np.zeros(
            (self.__number_of_gestures, self.__repetitions, self.__ch, self.__ch)
        )
        #       matrix where correlation coefficient matrices of each gesture repetition
        #       will be placed.
        for index in np.ndindex(self.__number_of_gestures, self.__repetitions):
            # loop for computing correlation coefficient matrices
            corr_package[index[0], index[1], :, :] = np.corrcoef(
                self.__gestures[index[0], index[1], :, :]
            )
        for gesture in range(self.__number_of_gestures):
            # loop for computing mean distances for matrices describing one gesture
            mean = mean_riemann(
                corr_package[gesture, :, :, :],
                tol=1e-08,
                maxiter=50,
                init=None,
                sample_weight=None,
            )
            self.__SPD_matrices[gesture, :, :] = mean[:, :]
        return self.__SPD_matrices

    def __str__(self):
        print("number of gestures in base: " + str(self.__number_of_gestures))
        print("number of repetitions for gesture: " + str(self.__repetitions))
        print("number of channels: " + str(self.__ch))
        return "number of samples for each gesture: " + str(self.__gestures.shape[3])

    def __getitem__(self, n):
        try:
            return self.__SPD_matrices[n, :, :]
        except IndexError:
            return (
                "please choose index from range [0,"
                + str(self.__number_of_gestures - 1)
                + "]"
            )

    def __repr__(self):
        return self.__str__()

    def add(self, other):
        if (
            self.__number_of_gestures != other.__number_of_gestures
            or self.__ch != other.__ch
            or self.__gestures.shape[3] != other.__gestures.shape[3]
        ):
            return ValueError(
                "Bases must have same number of gestures, channels and samples"
            )
        base_new = np.zeros(
            (
                self.__number_of_gestures,
                self.__repetitions + other.__repetitions,
                self.__ch,
                self.__gestures.shape[3],
            )
        )
        for i in range(self.__number_of_gestures):
            base_new[i, : self.__repetitions, :, :] = self.__gestures[i, :, :, :]
            base_new[i, self.__repetitions :, :, :] = other.__gestures[i, :, :]
        return base_new


class MDM(object):
    #   Accepts base made with class above and 2D matrix of gesture signals
    #   Returns index of gesture classified based on riemann geometry
    def __init__(self, base, gesture):  # base - 3d , gesture = 2D
        self.__base = base
        self.__gesture = gesture
        self.__number_of_gestures = base.shape[0]
        self.__distances = {}

    def classify(self):
        # computes distances between online gesture and each gesture from base
        corr_gesture = np.corrcoef(self.__gesture)
        for gesture in range(self.__number_of_gestures):
            self.__distances[gesture] = distance_riemann(
                self.__base[gesture, :, :], corr_gesture
            )
        return self.__distances

    def getmin(self, threshold=1):
        # chooses minimal distance (if bigger than threshold returns None)
        if 0 > threshold or threshold > 1:
            raise ValueError("Values between 0 and 1 only")
        elif min(self.__distances.values()) < threshold:
            return min(self.__distances, key=self.__distances.get)
        else:
            return None

    @property
    def number_of_gestures(self):
        return self.__number_of_gestures

    def __str__(self):
        for gesture, distance in self.__distances.items():
            print(
                "number of gesture: ",
                str(gesture),
                "         distance: ",
                str(distance),
            )
        return "  "

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, n):
        return self.__distances[n]
