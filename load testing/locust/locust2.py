import time
from locust import HttpUser, task, between

class MyUser(HttpUser):
    wait_time = between(1,5)
    host = "https://nle0yvx402.execute-api.us-east-1.amazonaws.com/beta"

    @task
    def launch_url(self):
        self.client.post(url="https://nle0yvx402.execute-api.us-east-1.amazonaws.com/beta",json=
        {
            "instances": [

                {
                    "features": [0,2,3,9,12,2020]
                }
            ]
        })

