import time
from locust import HttpUser, task, between

class WebsiteUser(HttpUser):
    wait_time = between(1,5)

    @task
    def index_page(self):
        self.client.post(url="https://nle0yvx402.execute-api.us-east-1.amazonaws.com/beta")

    @task
    def index_page(self):
        self.client.get(url="https://wfedxzgo2m.execute-api.us-east-1.amazonaws.com/beta")

