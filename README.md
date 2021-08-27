###CSC2227 Project

We developed an app to ask customers to query about a predicted Amazon EC2 price based on the input information. The predicted Amazon EC2 price would be either extracted from a pre-computing database or computed in the real-time machine learning component.

We have the below five parts in this repository:
(1) machine learning component: local machine learning model training and api design.
(2) pre computing database: connection with MySQL Sever and created a GET api.
(3) machine learning component cloud: Segamaker machine learning component.
(4) lambda function: POST function for cloud machine learning component and GET function for cloud pre computing database.
(5) load testing
