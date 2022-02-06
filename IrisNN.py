from sklearn import datasets
import random #acak
import math #sigmoid euler number
import matplotlib.pyplot as plt
# import some data to play with
iris = datasets.load_iris()
x = iris.data  # we only take the first two features.
y = iris.target

y_dupe = [] # 0 , 1 , 2
x_dupe = [] #make lists for each distinct value in y
index = [] #list of index for each distinct number
for i in range(0,len(y)):
    if y[i] not in y_dupe:
        y_dupe.append(y[i])
        index.append(i)

for i in x:
    x_dupe.append(i) #temporary list for x for alternating values
    
#index values for each target value
count = 0
for i in range(0,len(y)): #alternating between target values
    if count==0:
        y[i]=y_dupe[0]
    elif count==1:
        y[i]=y_dupe[1]
    elif count==2:
        y[i]=y_dupe[2]
    count = count + 1
    if count==3:
        count=0
        
x_ss = []  #alterntaing input
count=0
for i in range(0,int(len(x))):
    if count==0:
        x_ss.append(x_dupe[index[count]])
        index[count]+=1
    elif count==1:
        x_ss.append(x_dupe[index[count]])
        index[count]+=1
    elif count==2:
        x_ss.append(x_dupe[index[count]])
        index[count]+=1
    count = count + 1
    if count==3:
        count=0
          
x_norm = []
fitur1 = [] #make a list for each feature of the input
fitur2 = []
fitur3 = []
fitur4 = []
for i in x_ss:
    for j in range(0,4):
        if j == 0:
            fitur1.append(i[j])
        elif j == 1:
            fitur2.append(i[j])
        elif j == 2:
            fitur3.append(i[j])    
        elif j == 3:
            fitur4.append(i[j])

#find max and min for each feature          
maxf1=max(fitur1)
maxf2=max(fitur2)
maxf3=max(fitur3)
maxf4=max(fitur4)

minf1=min(fitur1)
minf2=min(fitur2)
minf3=min(fitur3)
minf4=min(fitur4)

#normalise x from 0-1
for i in x_ss:
    temp = []
    for j in range(0,4):
        if j == 0:
            temp.append((i[j]-minf1)/(maxf1-minf1))
        elif j == 1:
            temp.append((i[j]-minf2)/(maxf2-minf2))
        elif j == 2:
            temp.append((i[j]-minf3)/(maxf3-minf3))
        elif j == 3:
            temp.append((i[j]-minf4)/(maxf4-minf4))
    x_norm.append(temp)
    
fitur1 = [] #reinisialise for each feature as x has been normalised
fitur2 = []
fitur3 = []
fitur4 = []
for i in x_norm:
    for j in range(0,4):
        if j == 0:
            fitur1.append(i[j])
        elif j == 1:
            fitur2.append(i[j])
        elif j == 2:
            fitur3.append(i[j])    
        elif j == 3:
            fitur4.append(i[j])
            
average=[0]*4 #inisialise average and standard deviation lists
sd=[0]*4
for i in x_norm: #calculate average and standard edviation for each feature
    for j in range(0,4):
        if j == 0:
            average[j] = sum(fitur1)/len(x_norm)
            sd[j] = (sum((i[j] - average[j]) ** 2 for i in x_norm) / len(x_norm))**0.5
        elif j == 1:
            average[j] = sum(fitur2)/len(x_norm)
            sd[j] = (sum((i[j] - average[j]) ** 2 for i in x_norm) / len(x_norm))**0.5
        elif j == 2:
            average[j] = sum(fitur3)/len(x_norm)
            sd[j] = (sum((i[j] - average[j]) ** 2 for i in x_norm) / len(x_norm))**0.5
        elif j == 3:
            average[j] = sum(fitur4)/len(x_norm)
            sd[j] = (sum((i[j] - average[j]) ** 2 for i in x_norm) / len(x_norm))**0.5

z_score = [] #find z_score according to each feature's average and standard deviation
for i in x_norm:
    temp = []
    for j in range(0, len(i)):
        temp.append((i[j]-average[j])/sd[j])
    z_score.append(temp)

x_train = []
y_train = []
number = 0
number_prv = []
for i in range(0,99): #take training values from 1-99
    x_train.append(z_score[i])
    y_train.append(y[i])
x_test = []
y_test = []
#random choice of 10 values between 100-150
number = random.sample(range(100, len(x)), 10)
for i in number:   
    x_test.append(z_score[i])
    y_test.append(y[i])
    print(f'Index:{i}')

category = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1]} #one hot encoding
y_new_train = [category[i] for i in y_train]
y_new_test = [category[i] for i in y_test]

input_layer = 4
hidden_layer = 20
output_layer = 3

def weights(i):  #initisialise weights for v and w
    return -0.5 + random.random()
v = [[weights(i) for i in range(0, input_layer)] for l in range(0, hidden_layer)] 
w = [[weights(i) for i in range(0, hidden_layer)] for l in range(0, output_layer)]

#initialize beta for v and w
beta_v = 0.7 * (hidden_layer) ** (1/input_layer)
beta_w = 0.7 * (output_layer) ** (1/hidden_layer)

def normalizeV(x,y): #x->element y->sum dari list dalem list
    return beta_v*x/y

def normalizeW(x,y):
    return beta_w*x/y

def total(i):
    temp = []
    for j in i:
        temp.append(j**2) 
    return sum(temp)**0.5 #square root of the sum of elements squared

v_updt = [[normalizeV(i, total(sublist)) for i in sublist] for sublist in v]
w_updt = [[normalizeW(i, total(sublist)) for i in sublist] for sublist in w]

v0 = []
w0 = []

#bias inisialisation
for i in range(0, hidden_layer):
    v0.append(random.uniform(-beta_v, beta_v))
    
for i in range(0, output_layer):
    w0.append(random.uniform(-beta_w, beta_w))
    
alpha = 0 #momentum
mu = 0.1 #learning rate
deltaW_old = [[0]*hidden_layer]*output_layer 
deltaV_old = [[0]*input_layer]*hidden_layer
error_total = 1 #arbitrary value set intially for error
error_epoch = [] #error for each epoch
error_list = [] #error for the epoch
epoch = 0
threshold1 = 0.85 #>= than this = 1
threshold2 = 0.15 #<= than this = 0

while error_total > 0.01 and epoch < 1000: #epoch is commented only when training is solely dependent on error
    error_list = []
    for i in range(0, len(x_train)):
        xv = []
        z = []
        zy = []
        y = []
        ysigm = 0
        for k in range(0,len(v_updt)):
            temp = []
            for j in range(0, len(x_train[0])): #input * weights
                temp.append(x_train[i][j]*v_updt[k][j])
            xv.append(temp)
            z.append(1/(1+math.exp(-(sum(xv[k])+v0[k])))) #find output of hidden layer
            
        for k in range(0, len(w_updt)):
            temp = []
            for j in range(0, len(z)): #hidden * weights
                temp.append(z[j]*w_updt[k][j])
            zy.append(temp)
            ysigm = 1/(1+math.exp(-(sum(zy[k])+w0[k]))) #thresholds can be commented when epoch is commented
            if ysigm >= threshold1:
                ysigm = 1
            elif ysigm <= threshold2:
                ysigm = 0
            y.append(ysigm) #find output of output layer
        
        error = [(y_new_train[i][j]-y[j])**2 for j in range(0, len(y))] #error tk-yk^2
        error_per_set = sum(error)*0.5 #error for the set, sum dulu baru dikali 0.5
        error_list.append(error_per_set) #error buat tiap epoch dari tiap data set
        
        delta_k = [(y[j]-y_new_train[i][j])*(y[j]*(1-y[j])) for j in range(0, len(y))]
        deltaW = []
        deltaW0 = []
        for k in range(0, len(delta_k)):
            temp = []
            for j in range(0, len(z)):
                temp.append(mu*delta_k[k]*z[j]+alpha*deltaW_old[k][j]) #calculating deltaW
            deltaW.append(temp)
        deltaW_old = [[k[j] for j in range(0, len(k))] for k in deltaW]
        
        for k in range(0, len(w0)):
            deltaW0.append(alpha*delta_k[k]) #changing w bias
            w0[k] -= deltaW0[k]
        
        #calculation of error for input including delta j
        do_output = [sum([(delta_k[j]*w_updt[j][k]) for j in range(0, len(y))]) for k in range(0, len(w_updt[0]))]
        do_y = [z[j]*(1-z[j]) for j in range(0, len(z))]
        total_do_output = [[do_output[j]*do_y[j]*x_train[i][k] for k in range(0, len(x_train[0]))] for j in range(0, len(do_y))]
        
        deltaV = []
        deltaV0 = []
        for k in range(0, len(total_do_output)):
            temp = []
            for j in range(0, len(xv[0])):
                temp.append(mu*total_do_output[k][j]+alpha*deltaV_old[k][j]) #finding deltaV
            deltaV.append(temp)
        deltaV_old = [[k[j] for j in range(0, len(k))] for k in deltaV]
    
        for k in range(0, len(v0)):
            deltaV0.append(alpha*do_output[k]*do_y[k]) #change v bias
            v0[k] -= deltaV0[k]
        
        for k in range(0, len(v_updt)):
            for j in range(0, len(v_updt[0])): #change v according to deltaV
                v_updt[k][j] -= deltaV[k][j]
                
        for k in range(0, len(w_updt)):
            for j in range(0, len(w_updt[0])): #change w according to deltaW
                w_updt[k][j] -= deltaW[k][j]
    error_total = sum(error_list)/len(x_train) #find average error for each epoch, len(x_train) is commented only when average error is not used
    error_epoch.append(error_total) #store error for epoch
    epoch+=1

plt.plot(range(0, epoch), error_epoch) #plot the error vs epoch graph
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Plot of Error against Epoch')
s = (f'Momentum: {alpha},\nLearning Rate: {mu}') #string for text in graph
a = sum(error_epoch)/epoch #location for text
plt.text(0.5*epoch, a, s)
plt.show()

error_list_test = [] #testing feedforward
ysigm = 0
ycomp = []
for i in range(0, len(x_test)):
        xv = []
        z = []
        zy = []
        y = []
        for k in range(0,len(v_updt)):
            temp = []
            for j in range(0, len(x_test[0])): #input * weights
                temp.append(x_test[i][j]*v_updt[k][j])
            xv.append(temp)
            z.append(1/(1+math.exp(-(sum(xv[k])+v0[k])))) #find output of hidden layer
            
        for k in range(0, len(w_updt)):
            temp = []
            for j in range(0, len(z)): #hidden * weights
                temp.append(z[j]*w_updt[k][j])
            zy.append(temp)
            ysigm = 1/(1+math.exp(-(sum(zy[k])+w0[k])))
            if ysigm >= threshold1: #use threshold to get 1s and 0s
                ysigm = 1
            elif ysigm <= threshold2:
                ysigm = 0
            y.append(ysigm) #find output of output layer #find output of output layer
        print(y)
        ycomp.append(y)
        
        error = [(y_new_test[i][j]-y[j])**2 for j in range(0, len(y))] #error tk-yk^2
        error_per_set = sum(error)*0.5 #error for the set, sum dulu baru dikali 0.5
        error_list_test.append(error_per_set) #error buat tiap epoch dari tiap data set

correct = 0 #find correct and wrong outputs from the testing data
wrong = 0
for i in range(0, len(ycomp)):
    if set(ycomp[i]) == set(y_new_test[i]): #set compares data iteratively in order
        correct += 1
    else:
        wrong +=1
recog_rate = correct/len(ycomp)*100 #find recognition rate

print(f"Recognition rate: {recog_rate}, Number of epoch: {epoch}")