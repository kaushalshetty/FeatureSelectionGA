# FeatureSelectionGA
### Feature Selection using Genetic Algorithm (DEAP Framework)

Data scientists find it really difficult to choose the right features to get maximum accuracy especially if you are dealing with a lot of features. There are currenlty lots of ways to select the right features. But we will have to struggle if the feature space is really big. Genetic algorithm is one solution which searches for one of the best feature set from a lot of other features in order to attain a high accuracy.

#### Usage:
`model = LogisticRegression()
fsga = FearureSelectionGA(model,x_train,y_train)
pop = fsga.generate(100)
#Select the best individual from the final population and fit the initialized model
`



