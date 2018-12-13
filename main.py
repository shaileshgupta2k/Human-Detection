
import cv2 
import sys
from pathlib import Path
import numpy as np
import math
import os
import glob

# function to start initialization of the program
def main():

	# find path of the folder containing test and train images
	dataPath = sys.argv[1]

	# variables to store folder names for train images
	positiveTrainDataFolder = 'Train_Positive'
	negativeTrainDataFolder = 'Train_Negative'

	# variables to store folder names for test images
	positiveTestDataFolder = 'Test_Positive'
	negativeTestDataFolder = 'Test_Neg'

	# create variables to store path of images
	positiveTrainPath = dataPath + '/' + positiveTrainDataFolder
	negativeTrainPath = dataPath + '/' + negativeTrainDataFolder
	positiveTestPath = dataPath + '/' + positiveTestDataFolder
	negativeTestPath = dataPath + '/' + negativeTestDataFolder

	# create labels for images
	trainDataList = {positiveTrainPath: 1, negativeTrainPath: 0}
	testDataList = {positiveTestPath: 1, negativeTestPath: 0}

	# create object of class HistogramOfGradients
	hog = HistogramOfGradients()

	# function calls to find images for each of test and train
	trainImages, trainData = hog.findImages(trainDataList, '/')
	testImages, testData = hog.findImages(testDataList, '/')

	# compute HOG for each of test and train images
	trainDataIn = hog.HOGEntry(trainImages, False)
	testDataIn = hog.HOGEntry(testImages, True)

	# calls for neural network
	NN = NeuralNetwork()
	NN.trainNeuralNetwork(trainDataIn, trainData)
	NN.testNeuralNetwork(testImages, testDataIn, testData)


class HistogramOfGradients:

	# function to find all the images in a given folder
	def findImages(self, dataFile, delimeter):
		PathList = []
		dataOut = []

		for dataFolder in dataFile.keys():
			for directoryName, subDirectory, fileL in os.walk(dataFolder):
				for imageFile in fileL:
					imageP = dataFolder + delimeter + imageFile
					PathList.append(imageP)

					dataOut.append([dataFile[dataFolder]])

		return PathList, dataOut

	# read image from the given path
	def readImage(self, imagePath):
		return np.array(cv2.imread(imagePath, cv2.IMREAD_COLOR))

	# function to convert image from color to gray
	def colorToGray(self, img):

		# finding height, width and channels of image
		height, width, channels = img.shape
		i, j = 0, 0

		# create new image for storing gray scale values
		newImg = np.zeros((height, width))

		while i < height:
			j = 0
			while j < width:
				newImg[i][j] = round(0.299 * img[i][j][0] + 0.587 * img[i][j][1] + 0.114 * img[i][j][2])
				j = j + 1
			i = i + 1

		return newImg

	# function to compute HOG for the given image
	def HOG(self, imgG, imgTheta):

		# find histogram for each cell of the image
		cellHistogram = self.computeCellHistogram(imgG, imgTheta)
		cellHistogramSquared = np.square(cellHistogram)

		height, width = imgG.shape
		row, col = 0, 0
		cellRows = round(height/8)
		cellCols = round(width/8)
		HOGDescriptor = np.array([])

		# loops to create a block and normalize it and append the block to final output variable - HOGDescriptor
		while row < cellRows - 1:
			col = 0
			while col < cellCols - 1:

				block = np.array([])
				temp = np.array([])
				block = np.append(block,cellHistogram[row,col])
				block = np.append(block,cellHistogram[row,col+1])
				block = np.append(block,cellHistogram[row+1,col])
				block = np.append(block,cellHistogram[row+1,col+1])
				temp = np.append(temp,cellHistogramSquared[row,col])
				temp = np.append(temp,cellHistogramSquared[row,col+1])
				temp = np.append(temp,cellHistogramSquared[row+1,col])
				temp = np.append(temp,cellHistogramSquared[row+1,col+1])
				temp = np.sum(temp)
				if temp > 0:
					norm = np.sqrt(temp)
					block = (1/norm)*block
				HOGDescriptor = np.append(HOGDescriptor, block)
				col = col + 1
			row = row + 1

		return HOGDescriptor

	# ENTRY POINT function where all processing related to Gradient and HOG begins
	def HOGEntry(self, imagesPath, isPrint = False):

		#imagesFeatures = np.array([])
		imagesFeatures = []
		counter = 0

		for images in imagesPath:

			# function call to read image with given path
			inputColorImage = self.readImage(images)

			# convert image from color to grayscale
			grayImg = self.colorToGray(inputColorImage) 
			
			# function call to compute image gradient
			imgGradient, imgTheta = self.computeGradient(grayImg) #computing the gradient and the angle and using Prewitt's Operator

			# condition to write image gradient only if image is a test image
			if isPrint:
				cv2.imwrite(str(counter) + ".bmp", imgGradient)
				counter = counter + 1

			imageHOGDescriptor = self.HOG(imgGradient, imgTheta)
			np.set_printoptions(threshold=np.nan)

			# if images == "Human/Test_Positive/crop001045b.bmp":
			# 	with open('somefile.txt', 'a') as the_file:
			# 		for item in imageHOGDescriptor:
			# 			the_file.write("{}\n".format(item))

			imageHOGDescriptor = imageHOGDescriptor.reshape(-1,1)

			imagesFeatures.append(imageHOGDescriptor)

		return imagesFeatures


	# function to compute histogram for each cell
	def computeCellHistogram(self, imgG, imgTheta):

		# define cell size
		cellSize = 8

		# get height and width of the image
		height, width = imgG.shape

		#initialize the number of cell rows and cell columns
		cellRows = round(height/cellSize)
		cellCols = round(width/cellSize)

		# list to store histogram for each cell
		cellHistogram = np.zeros((cellRows,cellCols,9))

		i, j = 0, 0
		while i < cellRows - 1:
			j = 0
			while j < cellCols - 1:
				cellHistogram[i, j] = self.createHistogram(imgG, imgTheta, i, j, cellSize)
				j = j + 1
			i = i + 1
		return cellHistogram

	# function to compute histogram for a image cell
	def createHistogram(self, imgG, imgTheta, startI, startJ, cellSize):

		i, j = startI * 8, startJ * 8
		prev = 0
		cellHistogram = [0] * 9

		while i < startI * 8 + cellSize:
			j = startJ * 8
			while j < startJ * 8 + cellSize:

				angle = imgTheta[i][j]
				magnitude = imgG[i][j]

				# if angle is multiple of 20 then we straight away store value in the corresponding bin
				if angle % 20 == 0:
					if angle == 180:
						cellHistogram[0] += magnitude
					else:
						cellHistogram[int(angle / 20)] += magnitude
				else:
					prev = int(angle / 20)
					if angle > 160:
						cellHistogram[0] = cellHistogram[0] + ((180 - angle) / 20) * angle
					else:
						cellHistogram[prev + 1] = cellHistogram[prev + 1] + ((((prev + 1) * 20) - angle) / 20) * angle
						cellHistogram[prev] = cellHistogram[prev] + ((angle - (prev * 20)) / 20) * angle
				j = j + 1
			i = i + 1

		return cellHistogram

	# function to compute gradient for the given image
	def computeGradient(self, img):

		# Lists for storing prewitt's operator for Gx and Gy for finding gradient at each pixel location
		Gx = [[-1,0,1],
			  [-1,0,1],
			  [-1,0,1]]

		Gy = [[1,1,1],
			  [0,0,0],
			 [-1,-1,-1]]

		# getting rid of boundary constraint
		startI, startJ = 1, 1

		# finding height and width of image
		height, width = img.shape

		# create two lists for storing gradient values of image for X and Y; gradient theta and final gradient of image
		imgTheta = np.zeros((height, width))
		imgOut = np.zeros((height, width))

		# create i and j for iterating over the image
		i = startI
		j = startJ

		# loop for iterating over each pixel of the image and computing its X and Y gradient
		while i < height - startI:
			j = startJ
			while j < width - startJ:

				# finding gradient for X-axis i.e. Gx and finding the absolute value of gradient
				imgGX = self.convolveMatrices(img, Gx, i, j, int(len(Gx) / 2), int(len(Gx) / 2))
				if imgGX < 0:
					imgGX = abs(imgGX)

				# finding gradient for y-axis i.e. Gy and finding the absolute value of gradient
				imgGY = self.convolveMatrices(img, Gy, i, j, int(len(Gy) / 2), int(len(Gy) / 2))
				if imgGY < 0:
					imgGY = abs(imgGY)

				# normalize X and Y gradient
				imgGX = int(imgGX / 3.0)
				imgGY = int(imgGY / 3.0)


				#calculating gradient for final image and dividing the gradient by root of 2 to normalize the image to [0,255]
				imgOut[i][j] = np.sqrt(np.power(imgGX, 2)  + np.power(imgGY, 2)) / np.sqrt(2) # compute normalized gradient for whole image

				# if X gradient is zero set theta to 90 or -90 depending upon Y gradient
				if imgGX == 0:
					if imgGY > 0:
						imgTheta[i][j] = 90
					else:
						imgTheta[i][j] = -90
				else:
					imgTheta[i][j] = math.degrees(math.atan((imgGY / imgGX)))

				# if theta value is less than 0 then we need to round it to positive value to ease up the process of finding non-maxima suppression
				if imgTheta[i][j] < 0:
					imgTheta[i][j] = imgTheta[i][j] + 180

				# if imgTheta[i][j] >= 170:
				# 	imgTheta[i][j] = imgTheta[i][j] - 180

				if imgGX == 0 and imgGY == 0:
					imgOut[i][j] = 0
					imgTheta[i][j] = 0

				j = j + 1
			i = i + 1
		
		# return gradients of image and theta values of each pixel
		return imgOut, imgTheta

	# function to find convolution of two images
	def convolveMatrices(self, img, gradient, i, j, startI, startJ):

		#variables for iterating over gradient matrix
		gI, gJ = 0, 0

		#taking (i, j) as center of convolution, we shift left and up by half length of gradient
		i = i - startI
		j = j - startJ
		saveJ = j
		sum = 0

		# loop for multiplying sub-matrix with gaussian matrix over given [i-startI...i+startI, j-startJ...j+startJ] range
		while gI < len(gradient):
			gJ = 0
			j = saveJ
			while gJ < len(gradient[0]):
				sum = sum + (gradient[gI][gJ] * img[i][j])
				j = j + 1
				gJ = gJ + 1
			gI = gI + 1
			i = i + 1

		return sum


class NeuralNetwork:

	# class object initializer 
	# we are using 300 iterations and 1000 neurons for the hidden layer
	def __init__(self, graph = (7524, 1000, 1), ep = 200, lr = 0.01):

		self.graph = graph

		# store weights
		self.w1 = np.random.randn(graph[1], graph[0]) * 0.01
		self.w2 = np.random.randn(graph[2], graph[1]) * 0.01

		# store bias
		self.bias1 = np.zeros((graph[1], 1))
		self.bias2 = np.zeros((graph[2], 1))


		self.l1 = self.l2 = None

		self.d_w1 = self.d_w2 = None
		self.ep = ep
		self.lr = lr


	# function to implement feed forward algorithm
	def feedForward(self, trainDataSet):
		a1 = self.w1.dot(trainDataSet) + self.bias1
		self.l1 = self.ReLU(a1)
		self.l2 = self.sigmoid(self.w2.dot(self.l1) + self.bias2)

	# function to compute error
	def error(self, actual_output):
		return 0.5 * np.square(self.l2 - actual_output).sum()

	# function to do backpropagation
	def backpropagation(self, trainDataSet, actual_output):
		diff = self.l2 - actual_output
		z2 = diff * self.dSigmoid(self.l2)
		self.d_w2 = np.dot(z2, self.l1.T)

		z1 = np.dot(self.w2.T, z2) * self.dReLU(self.l1)
		self.d_w1 = np.dot(z1, trainDataSet.T)

		self.d_bias2 = np.sum(z2, axis = 1, keepdims = True)
		self.d_bias1 = np.sum(z1, axis = 1, keepdims = True)

	# function to update weights and bias values
	def update(self):
		self.w1 = self.w1 - self.lr * self.d_w1
		self.bias1 = self.bias1 - self.lr * self.d_bias1

		self.w2 = self.w2 - self.lr * self.d_w2
		self.bias2 = self.bias2 - self.lr * self.d_bias2


	# function to train neural network
	def trainNeuralNetwork(self, input_train_data, output_train_data):

		trainLen = len(input_train_data)
		for epoch in range(self.ep):
			epoch_error = 0.0
			for data_count, train_data in enumerate(input_train_data):
				self.feedForward(train_data)
				error = self.error(output_train_data[data_count])
				epoch_error += error
				self.backpropagation(train_data, output_train_data[data_count])
				self.update()

			print("Epoch Count: " + str(epoch), " | Average Error: ", epoch_error/trainLen)

	# function to test neural network
	def testNeuralNetwork(self, testImages, input_test_data, output_test_out):

		missedClassification = 0

		positiveList = []
		negativeList = []

		for data_count, test_data in enumerate(input_test_data):
			self.feedForward(test_data)
			print("Test Image: " + testImages[data_count] + " | Predicted Probability: " + str(self.l2) + " | Actual Probability: " + str(output_test_out[data_count]))

			cPrediction = np.round(self.l2.sum())

			if cPrediction:
				positiveList.append([testImages[data_count], str(self.l2.sum())])
			else:
				negativeList.append([testImages[data_count], str(self.l2.sum())])

			# finc error in detection
			missedClassification += (float(cPrediction - output_test_out[data_count]) == 0)


		print("Prediction Accuracy: " + str(float(missedClassification) / float(len(output_test_out)) * 100))

	# sigmoid function
	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	# derivate sigmoid function
	def dSigmoid(self, x):
		return x * (1 - x)

	def ReLU(self, t):
		return np.maximum(t, 0)

	def dReLU(self, t):
		return 1 * (t > 0)


if __name__ == "__main__":
	main()
