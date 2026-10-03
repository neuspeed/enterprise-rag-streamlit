import requests
import backoff
import json
import structlog
from requests.auth import HTTPBasicAuth

from settings import Settings
from utils.schemas import ReturnPayload

log = structlog.get_logger('webhook_sender')

class WebhookSender():
    DEFAULT_WEBHOOK = "https://{ENV_PATH}infercom.one/api/webhook/flowise-update"
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.webhook_url = self.DEFAULT_WEBHOOK.format(
            ENV_PATH=f"{self.settings.ENV}." if self.settings.ENV in ['dev','stage'] else "")


    @backoff.on_exception(backoff.expo, requests.exceptions.RequestException)
    def send_to_webhook(self, external_url, data: dict):
        headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
        basic = None
        if self.settings.ENV in ["stage"]:
            basic = HTTPBasicAuth(
                self.settings.STAGE_BASIC_AUTH_USER,
                self.settings.STAGE_BASIC_AUTH_PASS
            )
        
        response = requests.request("POST", external_url, headers=headers, data=data, auth=basic if basic else None)
        if response.status_code != 200:
            if response.status_code == 429:
                raise requests.exceptions.RequestException
        log.debug("send_to_webhook", status = response.status_code, reason=response.reason)
            