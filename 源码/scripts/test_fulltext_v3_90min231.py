"""Critical offline guards for the approved isolated real-data test."""
import inspect
import unittest
from unittest.mock import Mock,patch

from run_fulltext_v3_90min231 import OneShotTarget,pressure_reason
from benchmark_fulltext_source_id_isolated231_continuous import COLLECTION_PREFIX,SOURCE_COLLECTION
from fulltext_source_id_safe_common import IntegrityError


def healthy():
    return {'cpu_percent':10,'available_memory':900*1024**3,'data_free':20*1024**4,
            'target_memory':5*1024**3,'health':{str(p):{'status':200} for p in (9093,9094,9095,9099)}}


class Guards(unittest.TestCase):
    def test_pressure_is_protective_and_health_failure_stops_dispatch(self):
        self.assertIsNone(pressure_reason(healthy()))
        for key,value in [('cpu_percent',65),('available_memory',100*1024**3),
                          ('data_free',100*1024**3),('target_memory',106*1024**3)]:
            sample=healthy();sample[key]=value
            self.assertIsNotNone(pressure_reason(sample))
        sample=healthy();sample['health']['9094']={'error':'unavailable'}
        self.assertIsNotNone(pressure_reason(sample))

    def test_ambiguous_insert_is_attempted_once_and_error_is_propagated(self):
        raw=Mock();name=COLLECTION_PREFIX+'oneshot'
        writer=OneShotTarget(raw,name)
        with patch('run_fulltext_v3_90min231.inspect.unwrap',return_value=Mock(side_effect=TimeoutError('ambiguous'))) as unwrap:
            with self.assertRaises(TimeoutError):writer.insert(name,[{'id':'a'}])
            self.assertEqual(unwrap.return_value.call_count,1)
            raw.insert.assert_not_called()
        with self.assertRaises(IntegrityError):writer.insert(SOURCE_COLLECTION,[{'id':'a'}])
        self.assertFalse(hasattr(writer,'delete'))

    def test_installed_sdk_raw_insert_has_no_rpc_retry_loop(self):
        from pymilvus.client.grpc_handler import GrpcHandler
        raw=inspect.unwrap(GrpcHandler.insert_rows)
        code=inspect.getsource(raw)
        self.assertIn('self._stub.Insert(',code)
        self.assertNotIn('while ',code)
        self.assertIn('context',inspect.signature(raw).parameters)


if __name__=='__main__':unittest.main()
