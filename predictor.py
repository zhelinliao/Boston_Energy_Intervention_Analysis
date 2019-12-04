import nltk
import pickle
from collections import defaultdict
from LoadData import readFromFile
from nltk.probability import FreqDist, DictionaryProbDist, ELEProbDist




class EthnicityPredictor():
	
	def __init__(self):
		self._ethicity = ['am.ind.', 'asian', 'black', 'hispanic', 'white']
		self._fileData = readFromFile("surname_ethnicity_data.csv")

	def get3_let(self, name):
		if len(name) > 2:
			return [name[i:i+3] for i in range(len(name) - 2)]
		return [name]

	def buildFeatureList(self, data):
		features = set()
		
		for (name, stat, ethList) in data:
			x_lets = self.get3_let(name)
			for x_let in x_lets:
				features.add(x_let)

		self._featureList = sorted(features) #sorted(features.iteritems(), key=lambda (w,s): w, reverse=True) 	
		#number = [0, 1, 2, 3, 4]
		#self._mapEthic2Num = dict{zip(self._featureList, number)}
		self._featureNumber = len(features)
		print(self._featureNumber)

	
	def getFeature(self, name):
		#initVal = [False] * self._featureNumber
		#feature = dict(zip(self._featureList, initVal))
		x_lets = self.get3_let(name.lower())
		trueVal = [True] * len(x_lets)

		return dict(zip(x_lets, trueVal))
	
	'''
	def getFeatureWithLable(self, data):
		print(data[0])
		print(len(data))
		self.buildFeatureList(data)
		train_data = []
		cnt = 0;
		for (name, stat, ethList) in data:
			if stat < 100000:
				break
			cnt += 1
			if (cnt == 20):
				cnt = 0
				print(cnt, stat)
			
			for i in range(5):
				ethic = self._ethicity[i]
				num = int(ethList[i] / 1000)
				for j in range(num):
					train_data.append((self.getFeature(name), ethic))
		return train_data
	'''
	def saveTrainingResult2pkl(self, filename):
		file_train_result = open(filename, 'wb')
		pickle.dump(self.classifier, file_train_result, -1)

	def readProbabilityFromPkl(self, filename):
		pkl_file = open(filename, 'rb')
		self.classifier = pickle.load(pkl_file)


	def CreatNaiveBayes(self, data):
		

		label_freqdist = FreqDist() 
   
		for (name, total, ethList) in data:
			for i in range(5):
				label_freqdist[self._ethicity[i]] += ethList[i]

		label_probdist = ELEProbDist(label_freqdist) 
		feature_freqdist = defaultdict(FreqDist)
		feature_values = defaultdict(set)
		#for (name, total, ethList) in data:

		#	x-lets
		for (name, total, ethList) in data:
			x_lets = self.get3_let(name)
			for i in range(5):
				for x_let in x_lets:
					feature_freqdist[(self._ethicity[i], x_let)][True] += ethList[i]
					feature_values[x_let].add(True)

		for ((label, x_let), freqdist) in feature_freqdist.items():
			num = 0
			for i in range(5):
				if label == self._ethicity[i]:
					num = i
					break
			tot = 0
			for (name, total, ethList) in data:
				if x_let not in name:
					tot += ethList[num]
					feature_values[x_let].add(None)
			if tot > 0:
				feature_freqdist[(label, x_let)][None] += tot;
				

		feature_probdist = {}
		for ((label, fname), freqdist) in feature_freqdist.items():
			probdist = ELEProbDist(freqdist, bins=len(feature_values[fname]))
			feature_probdist[label, fname] = probdist
		
		self.classifier = nltk.NaiveBayesClassifier(label_probdist, feature_probdist)



	
	def TrainData(self):
		train_set = self._fileData[10000:]
		self.buildFeatureList(train_set)
		self.CreatNaiveBayes(train_set)
		self.saveTrainingResult2pkl('train_result.pkl')

		'''
		train_set = self.getFeatureWithLable(self._fileData)
		print("total :", len(train_set))
		time2 = time.time()
		print(time2 - time1)
		self.train(train_set)
		time3 = time.time()
		print(time3 - time2)
		file_train_result = open('train_result.pkl', 'wb')
		pickle.dump(self.classifier, file_train_result, -1)
		time4 = time.time()
		print(time4 - time3)
		'''

	def train(self, train_set):
		self.classifier = nltk.NaiveBayesClassifier.train(train_set)

	def classify(self, name):
        
		feature = self.getFeature(name)
		'''
		print(self.classifier.classify(feature))
		score = self.classifier.prob_classify(feature)
		#score = score.items()
		print('Probability:')
		for ethic in self._ethicity:
			print("%s : %.2f%% " % (ethic, score.prob(ethic) * 100))
        '''
		return self.classifier.classify(feature)


	def test(self, test_set):
		correct_cnt = 0

		for (name, total, ethList) in test_set:
			feature = self.getFeature(name)
			result = self.classifier.classify(feature)
			#score = self.classifier.prob_classify(feature)
			maxi = 0
			for i in range(5):
				if ethList[i] > ethList[maxi]:
					maxi = i
			if self._ethicity[maxi] == result:
				correct_cnt += 1

		print("accuracy : %.2f %%" % (correct_cnt / len(test_set) * 100))
		#return nltk.classify.accuracy(self.classifier, test_set)

		
#time1 = time.time()

predictor = EthnicityPredictor()
predictor.readProbabilityFromPkl('train_result.pkl')
#predictor.TrainData()
#print(predictor.get3_let("abcde"))
#predictor.test(predictor._fileData[:10000])
#print(predictor.classifier.labels())
#print(predictor.classifier.show_most_informative_features())
#print(predictor.classify("abcde"))

'''
dist = predictor.classifier.prob_classify({'abc': True})
for label in dist.samples():  
    print("%s: %f" % (label, dist.prob(label))) 
'''
'''
while 1 > 0:
	surname = input("Please input surname:\n")
	predictor.classify(surname)
'''