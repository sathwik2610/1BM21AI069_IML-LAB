import math as m

def mean(data):
    return sum(data) / len(data)

def var(data):
    sum_var = 0
    for i in range(len(data)):
        sum_var = sum_var + (data[i] - mean(data)) ** 2
    return sum_var

def cov(dt1, dt2):
    sum_cov = 0
    for i in range(len(dt1)):
        sum_cov += (dt1[i] - mean(dt1)) * (dt2[i] - mean(dt2))
    return sum_cov

physics = [15.0, 12.0, 8.0, 8.0, 7.0, 7.0, 7.0, 6.0, 5.0, 3.0]
history = [10.0, 25.0, 17.0, 11.0, 13.0, 17.0, 20.0, 13.0, 9.0, 15.0]

mean_physics = mean(physics)
mean_history = mean(history)

var_physics = var(physics)
var_history = var(history)

covariance = cov(physics, history)
std_physics = m.sqrt(var_physics)
std_history = m.sqrt(var_history)

r = covariance / (std_physics * std_history)
print(round(r, 3))
     
