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

class FeatureStore:
    """ Immutable abstract memory efficient interface to access feature info
    Indices returned from feature store can only be used with this instance of FeatureStore
    There can be many features, so you should not store results of API calls for large arrays

    usage examples:
    ft_traffic = [fidx for fidx in ft if 'generic' in fstore.get_flat_name(fidx)]
    ft_bgp = [fidx for fidx in ft if 'bgp' in fstore.get_encoding_path(fidx)]
    ft_bytes = np.array([fidx for fidx in ft if fstore.get_leaf_name(fidx) in ['bytes-sent', 'bytes-received']])
    # numpy array are also fine here we we only store unsigned integers as feature IDs

    to compare FeatureStores use `is_equal`

    avoid when possible for large arrays:
    names = [fstore.get_flat_name(fidx) for fidx in ft]

    ways for storing data:
    single process, single context: pyadt bindings
    multiple processes: deflated buffer
    """
    def __init__(self, fts):
        """ This is private. FeatureStore is created only when we receive data from flattener"""
        self._fts = fts

    def get_encpath(self, ftIdx):
        """ returns encoding path for feature """
        return self._fts[ftIdx].split('[')[0]

    def get_joined_kv(self, ftIdx):
        """ returns joined kv string( string between [ ] brackets without them """
        # we can to add a new API function in case we need to parse
        fn = self.get_flat_name(ftIdx)
        try:
            return fn.split('[')[1].split(']')[0]
        except IndexError:
            return ''

    def get_joined_path(self, ftIdx):
        """ returns nested path + leaf for feature joined with __  """
        return self._fts[ftIdx].split(']')[-1]

    def get_nested_path(self, ftIdx):
        """ returns nested path + leaf for feature joined with __  """
        return self.get_joined_path(ftIdx).split('__')

    def get_leaf(self, ftIdx):
        """ returns leaf path for feature. ex: bytes- """
        return self.get_joined_path(ftIdx).split('__')[-1]

    def get_flat_name(self, ftIdx):
        """ returns full feature name
        usually unique, but you shouldn't rely on that
        """
        return self._fts[ftIdx]

    def get_all_features(self):
        return list(range(len(self._fts)))

    def is_equal(self, fstore2):
        """ returns true if fstore2 has exactly the same feature set """
        return self is fstore2

    #def is_superset(self, fstore2):
    #    """ returns true if fstore2 is a suprtset of self """
    #    # we can extend fstore later to check if fstore2 only extends self
    #    return is_equal(fstore2)



if __name__ == "__main__":
    # usage example
    fts = [
            'Cisco-IOS-XR-ipv6-io-oper:ipv6-io/nodes/node/statistics/traffic[node-name=0/0/CPU0]icmp__sent-hop-count-expired-messages',
            'Cisco-IOS-XR-infra-statsd-oper:infra-statistics/interfaces/interface/latest/interfaces-mib-counters[interface-name=HundredGigE0/0/0/0]bytes-sent',
            'Cisco-IOS-XR-ipv4-io-oper:ipv4-network/nodes/node/statistics/traffic[node-name=0/0/CPU0]icmp-stats__hopcount-sent',
            ]
    fstore = FeatureStore(fts)
    all_fts = fstore.get_all_features()
    for ftIdx in all_fts:
        to_print = [
            ['ep', fstore.get_encpath(ftIdx)],
            ['leaf', fstore.get_leaf(ftIdx)],
            ['jpath', fstore.get_joined_path(ftIdx)],
            ['jkv', fstore.get_joined_kv(ftIdx)],
            ['flat', fstore.get_flat_name(ftIdx)],
        ]
        print(f"fstore[{ftIdx}]: ")
        for k, v in to_print:
            print(f"  {k:8} {v}")
    ft_hops_related = [ftIdx for ftIdx in all_fts if 'hop' in fstore.get_leaf(ftIdx)]
    print(f"fts with hops: {ft_hops_related}")

    print("Filtered features related to hops")
    for ftIdx in ft_hops_related:
        print(fstore.get_flat_name(ftIdx))
