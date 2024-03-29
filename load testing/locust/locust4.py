import time
from locust import HttpUser, task, between
import random

class MyUser(HttpUser):
    wait_time = between(1,5)
    host = "https://nle0yvx402.execute-api.us-east-1.amazonaws.com/beta"

    @task
    def launch_url(self):
        randomlist = []
        n1 = random.randint(0, 12)
        n2 = random.randint(0, 2)
        n3 = random.randint(0, 5)
        n4 = random.randint(1, 31)
        n5 = random.randint(1, 12)
        n6 = random.randint(2020, 2021)
        randomlist.append(n1)
        randomlist.append(n2)
        randomlist.append(n3)
        randomlist.append(n4)
        randomlist.append(n5)
        randomlist.append(n6)
        self.client.post(url="https://nle0yvx402.execute-api.us-east-1.amazonaws.com/beta",json=
        {
            "instances": [

                {
                    "features": randomlist
                }
            ]
        })

