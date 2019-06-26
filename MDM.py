# -*- coding: utf-8 -*-

"MDM"

""" TU SOM IMPORTY """
import numpy as np
import scipy
from scipy.linalg import eigvalsh
from numpy.core.numerictypes import typecodes


"""TU SOM DZIWNE RZECZY DO WAŻNYCH RZECZY"""
def _matrix_operator(Ci, operator):
    """matrix equivalent of an operator."""
    if Ci.dtype.char in typecodes['AllFloat'] and not np.isfinite(Ci).all():
        raise ValueError("Covariance matrices must be positive definite. Add regularization to avoid this error.")
    eigvals, eigvects = scipy.linalg.eigh(Ci, check_finite=False)
    eigvals = np.diag(operator(eigvals))
    Out = np.dot(np.dot(eigvects, eigvals), eigvects.T)
    return Out
def logm(Ci):
    return _matrix_operator(Ci, np.log)
def expm(Ci):
    return _matrix_operator(Ci, np.exp)
def sqrtm(Ci):
    return _matrix_operator(Ci, np.sqrt)
def invsqrtm(Ci):
    isqrt = lambda x: 1. / np.sqrt(x)
    return _matrix_operator(Ci, isqrt)
def _get_sample_weight(sample_weight, data):
    if sample_weight is None:
        sample_weight = np.ones(data.shape[0])
    if len(sample_weight) != data.shape[0]:
        raise ValueError("len of sample_weight must be equal to len of data.")
    sample_weight /= np.sum(sample_weight)
    return sample_weight

"""TU SOM WAŻNE RZECZY POTRZEBNE DO KLASYFIKATORA
    głównie matma z tej dokumentacji, przekopiowałam z dokumentacji pyriemanna to co potrzebne
    ... głównie dlatego że import za chuja pana mi nie działał i szybciej i łatwiej było tak"""
def mean_riemann(covmats, tol=10e-9, maxiter=50, init=None,
                 sample_weight=None):
    sample_weight = _get_sample_weight(sample_weight, covmats)
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

        crit = np.linalg.norm(J, ord='fro')
        h = nu * crit
        C = np.dot(np.dot(C12, expm(nu * J)), C12)
        if h < tau:
            nu = 0.95 * nu
            tau = h
        else:
            nu = 0.5 * nu

    return C

def distance_riemann(A, B):
    return np.sqrt((np.log(eigvalsh(A, B))**2).sum())
def dist_riem(A,B):
    return np.linalg.norm(scipy.linalg.logm(np.dot(np.linalg.inv(A) , B)))

"""Tutaj som właściwe perełki"""
class Base(object):
#   Ten Pan potrzebuje dostać 4-ro wymiarowego arraya z taką mapką indeksów:
#   (x,y,z,s) - to niech bedzie rozmiar
#       x - to będzie mówic ile mamy gestów
#       y - to mówi ile powtórzeń w każdym geście(musi ich być tyle samo dla każdego gestu)
#       z - tutaj mamy liczbe kanałów
#       s - liczba próbek sygnału jakie dostajemy (gługość sygnału wyrażona w próbkach)
#   Jeśli Pan dostanie wszystko co mu do szczęścia potrzebne to zwróci trójwymiarową macierz z średnimi odległościami
#   do której będziemy się odnosić przy porównywaniu sygnału/gestu live
    def __init__(self,arr):
        self.__gesty = arr
        self.__ilegestow = arr.shape[0]
        self.__ilepow = arr.shape[1]
        self.__ch = arr.shape[2]
        self.__SPDmatrices = np.zeros((self.__ilegestow,self.__ch,self.__ch))
    def Make_SPDBase(self):
        CovPackage = np.zeros((self.__ilegestow , self.__ilepow , self.__ch , self.__ch))
#       macierz do której będzieny wsadzać kowariancje pojedynczych gestów        
#        SPDmatrices = np.zeros((self.__ilegestow,self.__ch,self.__ch))
#       macierz 3d do której będziemy wkładać macierze 2d opisujące średnią odległość między macierzami kowariancji dla powtórzeń danego gestu
        for index in np.ndindex(self.__ilegestow,self.__ilepow):
            #w tej pętli liczmy macierz kowariancji wszystkich powtórzeń po kolei
            CovPackage[index[0],index[1],:,:] = np.corrcoef(self.__gesty[index[0],index[1],:,:])
        for gest in range(self.__ilegestow):
            # w tej pętli liczymy średnie odległości
            mean = mean_riemann(CovPackage[gest,:,:,:], tol=1e-08, maxiter=50, init=None, sample_weight=None)
            self.__SPDmatrices[gest,:,:] = mean[:,:]
        return self.__SPDmatrices
    def __str__(self):
        print('liczba gestow w bazie: '  + str(self.__ilegestow))
        print('każdy gest powtórzono: ' + str(self.__ilepow) + 'razy')
        print('liczba kanałów dla danych: ' +str(self.__ch))
        print(self.__SPDmatrices)
        return 'liczba probek sygnału dla pojedynczego gestu: ' + str(self.__gesty.shape[3])
    def __getitem__(self , n):
        try:
            return self.__SPDmatrices[n,:,:]
        except IndexError:
            return 'proszę podać indeks z zakresu [0,' + str(self.__ilegestow-1) + ']'
    def __repr__(self):
        return self.__str__()
    def add(self,other):
        if self.__ilegestow != other.__ilegestow or self.__ch != other.__ch or self.__gesty.shape[3] != other.__gesty.shape[3]:
            return ValueError('bazy muszą mieć te samą liczbę gestów, channeli i sampli')
        BazaNew = np.zeros((self.__ilegestow , self.__ilepow+other.__ilepow , self.__ch , self.__gesty.shape[3]))
        for i in range(self.__ilegestow):
            BazaNew[i,:self.__ilepow,:,:] = self.__gesty[i,:,:,:]
            BazaNew[i,self.__ilepow:,:,:] = other.__gesty[i,:,:,]
        return BazaNew
class MDM(object):
#   Ten Pan dostaje dwie rzeczy - baze stworzoną w sposób powyższy oraz wyciety gest z sygnalu/ow
#   Zwraca (jeśli będzie szczęśliwy) indeks gestu z bazy który jest najbliższy gestowi live (wegle geometrii riemanna)
    def __init__(self,baza,gest): #baza - 3d , gest = 3D
        self.__baza = baza
        self.__gest = gest
        self.__ilegestow = baza.shape[0]
        self.__wyniki = {}
    def classify(self):
        covGest = np.corrcoef(self.__gest)
        for gest in range(self.__ilegestow):
            self.__wyniki[gest] = distance_riemann(self.__baza[gest,:,:],covGest)
#            print(dist_riem(self.__baza[gest,:,:],covGest))
#            print(distance_riemann(self.__baza[gest,:,:],covGest))
        return self.__wyniki
    def getmin(self , threshold=1):
#        print(threshold)
#        print(min(self.__wyniki.values()))
        if 0>threshold or threshold >1:
            raise ValueError('Values between 0 and 1 only')
        elif min(self.__wyniki.values()) < threshold:
            return min(self.__wyniki, key = self.__wyniki.get)
        else:
            return None
    @property
    def number_of_gestures(self):
        return self.__ilegestow
    def __str__(self):
        print('nr gestu bazy     odległość do gestu live')
        for gest , odleglosc in self.__wyniki.items():
            print(str(gest) + '                 ' +str( odleglosc))
        return '  '
    def __repr__(self):
        return self.__str__()
    def __getitem__(self,n):
        return self.__wyniki[n]
'Tutaj sobie testuejmy tworzenie bazy na sygnale EKG'
# =============================================================================
#from  scipy.signal import freqz, group_delay, firwin, firwin2, butter, buttord, lfilter, filtfilt
#
#Fs = 2048
#ch = 3 # liczba kanałów 0 - N, 1 - P, 2 - L
#fin = open('beata1.raw', 'rb') 
#s = np.fromfile(fin, dtype='<f') 
#fin.close() 
#s = np.reshape(s,(len(s)//ch,ch)) # zmieniamy tablicę z jednowymiarowej na dwuwymiarową
##Filtrowanie
#b,a = butter(5, 1/(Fs/2), btype = 'highpass')
#d,c = butter(1, np.array([49,51])/(Fs/2), btype = 'bandstop')
#f,e = butter(5, 25/(Fs/2), btype = 'lowpass')
#
##2,1,0
#lewa = filtfilt(f,e,(filtfilt(d,c,(filtfilt(b,a,s[:,2])))))
#
#syg = np.array([lewa[0:Fs].T]) #trzeba transponowac, zeby indeksy sie zgadzaly
#
#
#N = np.zeros((7,10,ch,Fs*2))
#for i in range(7):
#    for j in range(10):
#        N[i,j,:,:] = s[90*(j+2)*(1+i):90*(i+1)*(j+2)+2*Fs,:].T
##print(N)
#
#
#'Robimy baze'
#Baza = Base(N)
#SPDBaza = Baza.Make_SPDBase() 
#
#'Robimy MDM'
#gest = s[20*Fs:22*Fs,:].T
#MDM = MDM(SPDBaza,gest)
#MDM.classify()
#print(MDM)
##print(Baza[MDM.getmin(0.1)])
#print(MDM.getmin())
# =============================================================================