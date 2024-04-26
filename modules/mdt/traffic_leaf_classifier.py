"""
Copyright (c) 2024 Cisco and/or its affiliates.
This software is licensed to you under the terms of the Cisco Sample
Code License, Version 1.1 (the "License"). You may obtain a copy of the
License at
               https://developer.cisco.com/docs/licenses
All use of the material herein must be in accordance with the terms of
the License. All rights not expressly granted by the License are
reserved. Unless required by applicable law or agreed to separately in
writing, software distributed under the License is distributed on an "AS
IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied.
"""

traffic_leafname = set([
            'bytes-received',
            'bytes-sent',
            'input-data-rate',
            'input-load',
            'input-packet-rate',
            'output-data-rate',
            'output-packet-rate',
            'output-load',
            'packets-received',
            'packets-sent',
            'packets-received',
            'total-packets',
            'generated-packets',
            'received-good-bytes',
            'received-good-frames',
            'received-total-bytes',
            'received-total-frames',
            'received-multicast-frames',
            'received-total-octet-frames-from1024-to1518',
            'received-total-octet-frames-from128-to255',
            'received-total-octet-frames-from1519-to-max',
            'received-total-octet-frames-from256-to511',
            'received-total-octet-frames-from512-to1023',
            'received-total-octet-frames-from65-to127',
            'received-total64-octet-frames',
            'received-unicast-frames',
            'received-broadcast-frames',
            'total-bytes-transmitted',
            'total-frames-transmitted',
            'total-bytes-received',
            'total-frames-received',
            'total-good-bytes-transmitted',
            'total-good-bytes-received',
            'transmitted-good-frames',
            'transmitted-multicast-frames',
            'transmitted-total-octet-frames-from1024-to1518',
            'transmitted-total-octet-frames-from128-to255',
            'transmitted-total-octet-frames-from1518-to-max',
            'transmitted-total-octet-frames-from256-to511',
            'transmitted-total-octet-frames-from512-to1023',
            'transmitted-total-octet-frames-from65-to127',
            'transmitted-total64-octet-frames',
            'transmitted-unicast-frames',
            'transmitted-broadcast-frames',
            'data-rates__input-data-rate',
            'data-rates__input-load',
            'data-rates__output-load',
            'data-rates__input-packet-rate',
            'data-rates__output-data-rate',
            'data-rates__output-packet-rate',
            'interface-statistics__full-interface-stats__bytes-sent',
            'interface-statistics__full-interface-stats__bytes-received',
            'interface-statistics__full-interface-stats__multicast-packets-received',
            'interface-statistics__full-interface-stats__multicast-packets-sent',
            'interface-statistics__full-interface-stats__packets-received',
            'interface-statistics__full-interface-stats__packets-sent',
            'interface-statistics__full-interface-stats__broadcast-packets-received',
            'interface-statistics__full-interface-stats__broadcast-packets-sent',
            'multicast-packets-sent',
            'multicast-packets-received',
            'ipv4-sent-packets',
            'ipv6-sent-packets',
            'ipv4-received-packets',
            'ipv6-received-packets',
            'ipv4-stats__packets-output',
            'ipv6-stats__packets-output',
            'send-packets-queued-net-io',
            'sent-packets-queued',
            'received-packets-queued-net-io',
            'received-packets-queued',
            'send-packets-queued',
            'sent-packets-queued',
            'broadcast-packets-received',
            'broadcast-packets-sent',
            'tcp-dropped-packets',
            'tcp-input-packets',
            'tcp-output-packets',
            'input-packets',
            'output-packets',
            'packets-output',
            'received-packets',
            'ipv4__received-multicast-packets',
            'ipv4__sent-multicast-packets',
            'ipv4__sent-redirect-messages',
            'ipv4__generated-packets',
            'ipv6__received-multicast-packets',
            'ipv6__sent-multicast-packets',
            'ipv6-node-discovery__sent-router-advertisement-messages',
            'ipv6-node-discovery__received-router-advertisement-messages',
            'ipv6-node-discovery__sent-neighbor-advertisement-messages',
            'ipv6-node-discovery__received-neighbor-advertisement-messages',
            'ipv6-node-discovery__sent-router-solicitation-messages',
            'ipv6-node-discovery__received-router-solicitation-messages',
            'ipv6-node-discovery__sent-neighbor-solicitation-messages',
            'ipv6-node-discovery__received-neighbor-solicitation-messages',
            'ipv6__sent-redirect-messages',
            'ipv6__generated-packets',
            'ipv6__total-packets',
            'ipv6__no-route-packets'
            ])


def traffic_leaf_test(lf: str) -> bool:
    if lf in traffic_leafname:
        return True
    else:
        return False
