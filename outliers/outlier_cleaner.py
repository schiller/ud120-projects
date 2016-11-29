#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    # import numpy as np
    n_rows = len(predictions)
    n_remain = int(n_rows * .9)

    errors = (net_worths - predictions) ** 2
    tuples = zip(ages, net_worths, errors)
    tuples.sort(key = lambda tuple: tuple[2])
    cleaned_data = tuples[0:n_remain]
    
    return cleaned_data