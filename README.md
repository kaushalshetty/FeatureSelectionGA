# FeatureSelectionGA
### Feature Selection using Genetic Algorithm (DEAP Framework)

Data scientists find it really difficult to choose the right features to get maximum accuracy especially if you are dealing with a lot of features. There are currenlty lots of ways to select the right features. But we will have to struggle if the feature space is really big. Genetic algorithm is one solution which searches for one of the best feature set from other features in order to attain a high accuracy.

#### Usage (Advanced):

By default, the FeatureSelectionGA has its own fitness function class. We can also define our own
FitnessFunction class.

```
class FitnessFunction:
    def __init__(self,n_splits = 5,*args,**kwargs):
        """
            Parameters
            -----------
            n_splits :int, 
                Number of splits for cv
            
            verbose: 0 or 1
        """
        self.n_splits = n_splits

    def calculate_fitness(self,model,x,y):
        pass
```

With this, we can design our own fitness function by defining our calculate fitness!
Consider the following example from Vieira, Mendoca, Sousa, et al. (2013)


Define the constructor __init__ with needed parameters:
```
class FitnessFunction:
    def __init__(self,n_total_features,n_splits = 5, alpha=0.01, *args,**kwargs):
        """
            Parameters
            -----------
            n_total_features :int
            	Total number of features N_t.
            n_splits :int, default = 5
                Number of splits for cv
            alpha :float, default = 0.01
                Tradeoff between the classifier performance P and size of 
                feature subset N_f with respect to the total number of features
                N_t.
            
            verbose: 0 or 1
        """
        self.n_splits = n_splits
        self.alpha = alpha
        self.n_total_features = n_total_features
```

```
    def calculate_fitness(self,model,x,y):
        alpha = self.alpha
        total_features = self.n_total_features

        cv_set = np.repeat(-1.,x.shape[0])
        skf = StratifiedKFold(n_splits = self.n_splits)
        for train_index,test_index in skf.split(x,y):
            x_train,x_test = x[train_index],x[test_index]
            y_train,y_test = y[train_index],y[test_index]
            if x_train.shape[0] != y_train.shape[0]:
                raise Exception()
            model.fit(x_train,y_train)
            predicted_y = model.predict(x_test)
            cv_set[test_index] = predicted_y
        
        P = accuracy_score(y, cv_set)
        fitness = (alpha*(1.0 - P) + (1.0 - alpha)*(1.0 - (x.shape[1])/total_features))
        return fitness
```

```
model = LogisticRegression()
fsga = FeatureSelectionGA(model,x_train,y_train)
pop = fsga.generate(100)
#Select the best individual from the final population and fit the initialized model
```



