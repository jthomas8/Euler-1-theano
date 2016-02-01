#!/usr/local/bin/python3
import theano

# Declare Theano symbolic variables
x = theano.tensor.dscalar('x')
y = theano.tensor.dscalar('y')

# Construct Theano expression graph
ap_limit = (y/x)  # takes the upper limit for the summation for arithmetic progression and removes the factor x which indicates the steps in the progression
ap_limit_int = (ap_limit - (ap_limit % 1)) # removes the fraction that may be left over from defining the upper limit
sum_ap_limit = ((ap_limit_int)*((ap_limit_int)+1)/2)  # Sums up the arithmetic progression
z = sum_ap_limit * x # Multiplies the factor back onto the summation of the progression

# Compile
f = theano.function(
        inputs = [x, y],
        outputs = z)
print(f(3, 999) + f(5, 999) - f(15, 999))
