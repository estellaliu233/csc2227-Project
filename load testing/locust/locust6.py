from numpy.random import choice
from locust import HttpUser, task, between
import random

class MyUser(HttpUser):
    wait_time = between(1,5)
    host = "https://nle0yvx402.execute-api.us-east-1.amazonaws.com/beta"

    @task
    def launch_url(self):
        randomlist = []
        randomlist2=[]
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
        j1 = random.randint(0, 2)
        j2 = random.randint(0, 2)
        j3 = random.randint(0, 5)
        j4 = random.randint(1, 31)
        j5 = random.randint(1, 12)
        j6 = random.randint(2020, 2021)
        randomlist2.append(j1)
        randomlist2.append(j2)
        randomlist2.append(j3)
        randomlist2.append(j4)
        randomlist2.append(j5)
        randomlist2.append(j6)
        candidates = [randomlist,randomlist2]
        draw = random.choices(candidates,weights=[0.3,0.7],k=1)
        for i in draw:
            draw2 = i
        self.client.post(url="https://nle0yvx402.execute-api.us-east-1.amazonaws.com/beta",json=
        {
            "instances": [

                {
                    "features": draw2
                }
            ]
        })

