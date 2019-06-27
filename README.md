# MDM
Part of a bigger project. Allows to characterise signals collected offline as 2D arrays. With such characterisation it is possible to classify online signals.

class Base(arr):

	arr - 4D array with shape (x,y,z,s), where x - number of gestures, y - number of trials for gesture (same for every gesture), z - number of channels, s - number of samples

method Base.Make_SPDBase():
	
	returns:
	3D array of characterised gestures (name it classes) with shape (number of gestures , number of channels, number of channels).

class MDM(base,gesture):
	
	base - base made with Base.Make_SPDBase()
	gesture - 2d array of gesture, must be same shape as gestures in Base(not SPDBase, raw Base)

method MDM.classify():
	
	returns:
	dictionary, where keys - indexes of gesture, values - distances of gesture to classes(values between 0 and 1).

method MDM.getmin(threshold = 1):
	returns:
	-index of class which distance is minimum to gesture.
	-None if minimum distance is bigger than threshold 
	Threshold must be between 0 and 1.  
	
