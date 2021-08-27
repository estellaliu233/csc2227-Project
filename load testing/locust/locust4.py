import time
from locust import HttpUser, task, between

class MyUser(HttpUser):
    wait_time = between(1,5)
    host = "https://wfedxzgo2m.execute-api.us-east-1.amazonaws.com/beta/Price_Database"

    @task
    def launch_url(self):
        self.client.post(url="https://wfedxzgo2m.execute-api.us-east-1.amazonaws.com/beta/Price_Database",json=
        {
            "instances": [

                {
                    "features": [0,2,3,9,12,2020]
                }
            ]
        })