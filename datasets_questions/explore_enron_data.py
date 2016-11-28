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

print len(enron_data)

# print len(enron_data["SKILLING JEFFREY K"])

# poi_count = 0
# for person in enron_data:
#   poi_count += enron_data[person]["poi"]
# print poi_count

# print enron_data["PRENTICE JAMES"].keys()
# print enron_data["PRENTICE JAMES"]["total_stock_value"]

# print enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]

# print enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]

# print enron_data.keys()
# print enron_data["SKILLING JEFFREY K"]["total_payments"]
# print enron_data["LAY KENNETH L"]["total_payments"]
# print enron_data["FASTOW ANDREW S"]["total_payments"]

# salary_count = 0
# email_count = 0
# for key in enron_data:
# 	if enron_data[key]["salary"] != "NaN":
# 		salary_count += 1
# 	if enron_data[key]["email_address"] != "NaN":
# 		email_count += 1
# print salary_count
# print email_count

total_payments_count = 0
for key in enron_data:
	if enron_data[key]["total_payments"] == "NaN":
		total_payments_count += 1
print total_payments_count
# print float(total_payments_count) / len(enron_data)

# poi_count = 0
# poi_payments_nan = 0
# for key in enron_data:
# 	if enron_data[key]["poi"]:
# 		poi_count += 1
# 		if enron_data[key]["total_payments"] == "NaN":
# 			poi_payments_nan += 1
# print float(poi_payments_nan) / poi_count