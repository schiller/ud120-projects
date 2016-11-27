#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

# print len(enron_data)

# print len(enron_data["SKILLING JEFFREY K"])

# poi_count = 0
# for person in enron_data:
#   poi_count += enron_data[person]["poi"]
# print poi_count

# print enron_data["PRENTICE JAMES"].keys()
# print enron_data["PRENTICE JAMES"]["total_stock_value"]

# print enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]

# print enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]

print enron_data["SKILLING JEFFREY K"]["bonus"]
print enron_data["LAY KENNETH"]["bonus"]
print enron_data["FASTOW ANDREW"]["bonus"]