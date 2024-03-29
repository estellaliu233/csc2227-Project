import random
from locust import HttpUser, task, between

class MyUser(HttpUser):
    wait_time = between(1,5)
    host = "https://cx5rbaz5g7.execute-api.us-east-1.amazonaws.com/default/lambdards"

    @task
    def launch_url(self):
        n1 = str(random.randint(0, 12))
        n2 = str(random.randint(0, 2))
        n3 = str(random.randint(0, 5))
        n4 = str(random.randint(1, 31))
        n5 = str(random.randint(1, 12))
        n6 = str(random.randint(2020, 2021))
        ts=[("Location",n1),("Instance",n2),("OS",n3),("day",n4),("month",n5),("year",n6)]
        features = dict(ts)
        self.client.post(url="https://cx5rbaz5g7.execute-api.us-east-1.amazonaws.com/default/lambdards",json=
        features
        )

