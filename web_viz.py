import json

from websockets.sync.client import connect

WS_ADDRESS = "wss://transdimensional.xyz/broadcast"


class WebViz:
    def __init__(self, upload_interval=500):
        self.websocket = connect(WS_ADDRESS)
        self.upload_interval = upload_interval
        self.coord_list = []

    def broadcast_position(self, position):
        self.coord_list.append(position)

        if len(self.coord_list) < self.upload_interval:
            return

        msg = {
            "metadata": {
                "user": "org3",
            },
            "coords": self.coord_list,
        }

        self.coord_list = []

        self.websocket.send(json.dumps(msg))
