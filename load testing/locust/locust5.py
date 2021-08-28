import random
from locust import HttpUser, task, between

class MyUser(HttpUser):
    wait_time = between(1,5)
    host = "http://127.0.0.1:5000"

    @task
    def launch_url(self):
        n1 = str(random.randint(0, 12))
        n2 = str(random.randint(0, 2))
        n3 = str(random.randint(0, 5))
        n4 = str(random.randint(1, 31))
        n5 = str(random.randint(1, 12))
        n6 = str(random.randint(2020, 2021))
        ts= "http://127.0.0.1:5000/emp/{}/{}/{}/{}/{}/{}".format(n1,n2,n3,n4,n5,n6)

        self.client.get(url=ts)

