import numpy as np
import copy


def periodic_kernel(x_1,x_2,sigma,timescale,period):
    result= sigma**2*np.exp(-(2/timescale**2)*np.sin(np.pi*np.abs(x_1-x_2)/period)**2)
    return result

def white_noise_kernel(x_1,x_2,sigma):
    # is statement checks if the variable x_1 references the same object with x_2. Not ==
    result = sigma**2 if x_1 is x_2 else 0
    return result

def exp_quadratic_kernel(x_1,x_2,sigma,timescale):
    result= sigma**2*np.exp(-(x_1-x_2)**2/(2*timescale**2))
    return result


class Normal():
    def __init__(self,mean,variance):
        super(Normal,self).__init__()

        self.mean=mean
        self.variance=variance

    def sample(self,size):
        result=np.random.Generator.normal(self.mean,np.sqrt(self.variance),size)
        return result

    def log_prob(self,x):
        result=np.log(1/(np.sqrt(np.pi*self.variance)))-(x-self.mean)**2/2
        return result


class MultivariateNormal():
    def __init__(self,mean,covariance):
        super(MultivariateNormal,self).__init__()

        self.mean=mean
        self.covariance=covariance

    def sample(self,size):
        result=np.random.multivariate_normal(self.mean,self.covariance,size)
        return result

    def log_prob(self,x):
        dim=len(self.mean)
        result=np.log(1/(np.sqrt(np.pi**dim)*np.linalg.det(self.covariance)))-((x-self.mean).T@self.covariance@(x-self.mean))/2
        return result



class GaussianProcess():
    def __init__(self, cov_func, prior=Normal(0,1)):
        super(GaussianProcess,self).__init__()

        self.kernel=cov_func
        self.joint_dist=prior
        self.prior_dist=prior
        # {x_1:[y_11,y_12,y_13],x_2:[y_2],...}
        self.data=dict()

    def vmean_func(self,xs_star):

        def mean_func(x_star):
            xs=self.data.keys()
            # return existing point mean
            if x_star in keys:
                return np.array(self.data[x]).mean()
            # intrapolate/extrapolate
            else:
                # K_22=np.array([[self.kernel(x_star,x_star)]])
                K_21=np.array([[self.kernel(x_star,x_j) for x_j in xs]])
                K_12=K_21.T
                K_11=self.joint_dist.covariance
                return self.prior_dist.mean+K_21@np.linalg.inv(K_11)@(np.array(self.data[x]).mean()-self.prior_dist.mean)

        vmean_func=np.vectorize(mean_func)
        return vmean_func


    def add_data(self,xs,ys):
        keys=self.data.keys()
        for x,y in zip(xs,ys):
            if x in keys:
                self.data[x].append(y)
            else:
                self.data[x]=[y]


    def remove_data(self,xs,ys):
        keys=self.data.keys()
        for x,y in zip(xs,ys):
            if x in keys:
                ys_x=self.data[x]
                ys_x.remove(y)
                if len(ys_x)==0:
                    del self.data[x]
            else:
                pass


    def update_joint(self,xs,ys):
        xs_old=copy.copy(list(self.data.keys()))
        filter=lambda x:x not in xs_old
        vfilter=np.vectorize(filter)
        mask=vfilter(xs)
        xs_star=xs[mask]

        # We need to average all the observations for common x values in xs and old_xs. We use the data dict for this.
        self.add_data(xs,ys)
        mean_star=self.vmean_func(xs)

        # Avoid recomputing te kernel for unchanged indices
        # covariance_star=np.array([[self.kernel(x_i,x_j) for x_j in xs] for x_i in xs])

        K_22=np.array([[self.kernel(x_i,x_j) for x_j in xs_star] for x_i in xs_star])
        K_21=np.array([[self.kernel(x_i,x_j) for x_j in xs_old] for x_i in xs_star])
        K_12=K_21.T
        K_11=np.array(self.joint_dist.variance) if isinstance(self.joint_dist,Normal) else self.joint_dist.covariance
        covariance_star=np.vstack((np.hstack((K_11,K_12)),np.hstack((K_21,K_22))))

        if isinstance(self.joint_dist,Normal):
            self.joint_dist=MultivariateNormal(mean_star,covariance_star)
        else:
            self.joint_dist.mean=mean_star
            self.joint_dist.covariance=covariance_star


    def infer(self,xs_star):
        xs=self.data.keys()
        predictive_dist=self.get_marginal(xs_star,xs,self.joint_dist.mean)

        return predictive_dist


    # xs_star may include either x from xs or x_new from xs_new
    # latter two arguments are the indices marginalized out/conditioned on.
    # def get_marginal(self,xs_new,ys_new,xs,ys):
    def get_marginal(self,xs_star,xs,ys):

        K_22=np.array([[self.kernel(x_i,x_j) for x_j in xs_star] for x_i in xs_star])
        K_21=np.array([[self.kernel(x_i,x_j) for x_j in xs] for x_i in xs_star])
        K_12=K_21.T
        K_11=self.joint_dist.covariance

        conditional_mean_star=self.vmean_func(xs_star)+K_21@np.linalg.inv(K_11)@(ys-self.vmean_func(xs))
        conditional_covariance_star=K_22-K_21@np.linalg(K_11)@K_21.T

        marginal_dist=MultivariateNormal(conditional_mean_star,conditional_covariance_star)

        return marginal_dist





# what size to use for the prior? 5?
# p=Normal(np.zeroes(5),np.diag(np.ones(5)))
# k=alpha_1*periodic_kernel(_,_,90)+alpha_2*periodic_kernel(_,_,12)+alpha_3*periodic_kernel(_,_,3)+exp_quadratic_kernel(_)+white_noise_kernel(_)
# GP(_,kernel=k,prior=p)
