from aeon.classification.base import BaseClassifier
import pyriemann
import numpy as np

class RiemannInterval(BaseClassifier):

    _tags = {
        "capability:multivariate": True,
    }
    def __init__(self,
                 cross_validate=False,
                 use_convexity=True,
                 use_surprise=False,
                 intervals=5,
                 scale=2,
                 random_state=None
            ):
        self.cross_validate=cross_validate
        self.use_convexity=use_convexity
        self.use_surprise=use_surprise
        self.intervals=intervals
        self.scale=scale
        self.random_state=random_state
        self.sfreq=None
        super().__init__()


    def _fit(self,X,y):
        self.means=[]
        self.convexity_weights = []
        self.surprise_weights = []
        self.classes_=np.unique(y)
        convexity_measure=[]
        if self.cross_validate:
            from sklearn.model_selection import GridSearchCV
            parameters = {"intervals":list(range(3,7,2)),"scale":list(range(1,10))}
            cls = IntervalRiemannianEnsemble(cross_validate=False)
            clf = GridSearchCV(cls,parameters)
            clf.fit(X,y)
            self.intervals=clf.best_params_["intervals"]
            self.scales = clf.best_params_["scale"]

        remainder = int(len(X[0][0])%self.intervals)
        if self.intervals==1:
            self.use_convexity=False
            self.use_surprise=False
        if remainder!=0:
            intervals = np.array_split(X[:,:,:-remainder],self.intervals,axis=2)
        else:
            intervals = np.array_split(X,self.intervals,axis=2)
        if self.use_surprise:
            self.surprise_weights=surprise_calculation(intervals,y)
        for i in range(len(intervals)):
            interval_mean=[]
            X_cov_train = pyriemann.utils.covariance.covariances(intervals[i],estimator="lwf")
            self.fg = pyriemann.tangentspace.FGDA()
            X_cov_train = self.fg.fit_transform(X_cov_train,y)
            for class_val in self.classes_:
                idx = np.where(y==class_val)
                slice_X_class = X_cov_train[idx[0]]
                class_mean = pyriemann.utils.mean.mean_riemann(slice_X_class)
                interval_mean.append(class_mean)
            self.means.append(interval_mean)
            if self.use_convexity:
                convexity_measure.append(calculate_convexity(intervals[i],y))
        if self.use_convexity:
            self.convexity_weights = [(1-x) for x in convexity_measure]



    def _predict_proba(self, X):
        X_test_distance=[]
        intervals = np.array_split(X,self.intervals,axis=2)
        interval_covs = []
        for j in range(len(intervals)):
            X_cov_test=pyriemann.utils.covariance.covariances(intervals[j],estimator="lwf")
            X_cov_test = self.fg.transform(X_cov_test)
            interval_covs.append(X_cov_test)


        for i in range(len(X)):
            interval_dist=[]
            for j in range(len(intervals)):  
                distances=[]
                count=0
                for r in range(-1,2,1): 
                    if (j+r)>-1 and (j+r)<self.intervals:
                        count+=1
                        for c in range(len(self.classes_)):
                            distances.append(pyriemann.utils.distance.distance_riemann(interval_covs[j][i],np.array(self.means[j+r][c])))
                for c in range(len(self.classes_)):
                    distances[c]=distances[c]/count
                interval_dist.append(distances)
            X_test_distance.append(interval_dist)
        probs=[]
        if self.use_convexity and self.use_surprise:
                alpha = 0.75
                beta = 0.25
                weights = (np.array(self.convexity_weights)**alpha)*(np.array(self.surprise_weights)**beta)
                t = 1.5
                s=5
                penalty = 1 / (1 + np.exp(s * (np.array(self.surprise_weights) - t)))
                weights = list(weights*penalty)
                weights = [int((x)*100) for x in weights]
                weights=scale_values_np(weights,self.scale)
        elif self.use_convexity:
            weights = [int((x)*100) for x in np.array(self.convexity_weights)]
            weights=scale_values_np(weights,self.scale)
        elif self.use_surprise:
            weights = [int((x)*100) for x in np.array(self.surprise_weights)]
            weights=scale_values_np(weights,self.scale)
        else:
            weights = [10 for x in range(self.intervals)]
        for inst in range(len(X_test_distance)):
            inst_votes=[]
            instance=X_test_distance[inst]
            count=0
            for interval in range(len(instance)):
                for w in range(weights[interval]):
                    smallest = (np.argmin(instance[interval])%len(self.classes_))
                    temp = list(np.zeros(len(self.classes_)))
                    temp[smallest]=1
                    inst_votes.append(temp)
                    count+=1
            
            counts=np.sum(inst_votes,axis=0)

            for i in range(len(counts)):
                counts[i]=float(counts[i]/count)
            probs.append(list(counts))


        return np.array(probs)
    
    def _predict(self, X):
        probas = self._predict_proba(X)
        idx = np.argmax(probas, axis=1)
        preds = np.asarray([self.classes_[x] for x in idx])
        return preds
    
def scale_values_np(arr,scale):
    arr = np.asarray(arr)
    n = len(arr)
    ranks = np.argsort(np.argsort(arr))
    min_scale = 1/scale
    max_scale = scale
    factors = min_scale + (max_scale - min_scale) * ranks / (n - 1)
    return (arr * factors).astype(int)


def reconstruct_path(predecessors, start, end):
    path = []
    i = end
    while i != start:
        if i == -9999:
            return None 
        path.append(i)
        i = predecessors[start, i]
    path.append(start)
    return path[::-1]


def surprise_calculation(intervals,y):
    from aeon.regression.convolution_based import MiniRocketRegressor
    import random
    n_intervals,n_instances,n_channels,n_timepoints= np.shape(intervals)
    scores=[0.4]
    for inst in range(0,n_intervals-1):
        insts_mse = [] 
        picks=[-1]
        for i in range(int(n_instances/10)):
            pick=-1
            while pick in picks:
                pick = random.randint(0,n_instances-1)
            picks.append(pick)

            train_index = [x for x in range(n_instances) if x!=pick]
            X_trains = intervals[inst][train_index]
            y_trains = intervals[inst+1][train_index]
            X_tests = intervals[inst][pick]
            y_tests = intervals[inst+1][pick]
            mse = []
            for j in range(n_channels):
                X_train = X_trains[:,j,:]
                y_train=np.mean(y_trains[:,j,:],axis=1)
                X_test = X_tests[j,:]
                y_test = np.mean(y_tests[j,:])
                rgr = MiniRocketRegressor()
                rgr.fit(X_train,y_train)
                pred = rgr.predict(X_test[np.newaxis,:])[0]
                mse.append((y_test-pred)**2)
            insts_mse.append(np.mean(mse))
        scores.append(np.mean(insts_mse))
    scores[0] = np.mean(scores[1:])
    return scores



def calculate_convexity(X,y):
    distance_matrix = np.zeros((len(X),len(X)))
    covariances = pyriemann.utils.covariance.covariances(X,estimator="lwf")
    from scipy.linalg import logm
    from scipy.spatial.distance import pdist, squareform
    logs = np.array([logm(cov) for cov in covariances])
    flat = logs.reshape(logs.shape[0], -1)
    distance_matrix = squareform(pdist(flat, metric='euclidean'))
    from scipy.sparse.csgraph import dijkstra
    

    k=3
    new_matrix = np.full_like(distance_matrix, np.inf)
    for i in range(len(X)):
        row = distance_matrix[i].copy()
        row[i] = np.inf
        k_indices = np.argpartition(row, k)[:k]
        for idx in k_indices:
            new_matrix[i, idx] = distance_matrix[i, idx]
    dist,pres = dijkstra(new_matrix,return_predecessors=True)
    score=[]
    for c in np.unique(y):
        idx = np.where(y==c)[0]
        cscore=[]
        for start in idx:
            for end in idx:
                if start!=end:
                    path =reconstruct_path(pres,start,end)
                    if path is None:
                        cscore.append(1)
                    else:
                        temp=0
                        for i in path:
                            if y[i]!=c:
                                temp+=1
                        cscore.append(float(temp/len(path)))
        score.append(np.mean(cscore))
    return np.mean(score)

    