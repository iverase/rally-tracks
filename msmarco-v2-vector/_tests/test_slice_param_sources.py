# Licensed to Elasticsearch B.V. under one or more contributor
# license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright
# ownership. Elasticsearch B.V. licenses this file to you under
# the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import importlib.util
import json
import pathlib
import random
from unittest import mock

_TRACK_PY = pathlib.Path(__file__).parents[1] / "track.py"
_spec = importlib.util.spec_from_file_location("msmarco_v2_vector_track_slice", _TRACK_PY)
track_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(track_module)


class StaticTrack:
    indices = [mock.Mock(name="msmarco-v2")]


class TestBulkSliceParamSource:
    def test_exposes_corpora_for_rally_download(self, monkeypatch):
        inner = mock.Mock()
        inner.infinite = False
        inner.corpora = [mock.Mock(name="msmarco-v2_base64-initial-indexing-1")]
        monkeypatch.setattr(track_module, "BulkIndexParamSource", lambda track, params, **kwargs: inner)
        source = track_module.BulkSliceParamSource(StaticTrack(), {"bulk-size": 100})
        assert source.corpora is inner.corpora

    def test_adds_random_slice_to_bulk_action(self):
        inner = mock.Mock()
        inner.infinite = False
        inner.params.return_value = {
            "body": '{"index":{"_index":"msmarco-v2"}}\n{"docid":"1"}\n'
        }
        partition = track_module._SliceRewritingPartition(inner, random.Random(42))
        params = partition.params()
        action = json.loads(params["body"].split("\n")[0])
        assert action["index"]["slice"] == str(random.Random(42).randint(0, 9999))


class TestBulkCopyDocIdSliceParamSource:
    def test_exposes_corpora_for_rally_download(self, monkeypatch):
        inner = mock.Mock()
        inner.infinite = False
        inner.corpora = [mock.Mock(name="msmarco-v2_base64-initial-indexing-1")]
        monkeypatch.setattr(track_module, "BulkIndexParamSource", lambda track, params, **kwargs: inner)
        source = track_module.BulkCopyDocIdParamSource(StaticTrack(), {"bulk-size": 100})
        assert source.corpora is inner.corpora

    def test_adds_docid_and_slice_to_bulk_action(self):
        source = track_module.BulkCopyDocIdParamSource(StaticTrack(), {})
        inner = mock.Mock()
        inner.infinite = False
        inner.partition.return_value = inner
        inner.params.return_value = {
            "body": '{"index":{"_index":"msmarco-v2"}}\n{"docid":"00_1"}\n'
        }
        source._inner = inner
        params = source.partition(0, 1).params()
        lines = [line for line in params["body"].split("\n") if line]
        action = json.loads(lines[0])
        assert action["index"]["_id"] == "00_1"
        assert action["index"]["slice"].isdigit()


class TestKnnParamSourceSlice:
    def test_adds_random_slice_request_param(self, monkeypatch):
        monkeypatch.setattr(
            track_module,
            "random_slice_request_params",
            lambda params: {"slice": "1234"},
        )
        source = track_module.KnnParamSource.__new__(track_module.KnnParamSource)
        source._index_name = "msmarco-v2"
        source._params = {"random-query": True, "dims": 1024}
        source._random_query = True
        source._dims = 1024
        source._iters = 0
        source._maxIters = 1
        source._queries = []

        params = source.params()
        assert params["request-params"] == {"slice": "1234"}


class TestRandomSliceRequestParams:
    def test_disabled_when_slice_enabled_is_false(self):
        assert track_module.random_slice_request_params({"slice_enabled": False}) == {}

    def test_uses_fixed_search_slice_id_when_set(self):
        assert track_module.random_slice_request_params({"search_slice_id": 4821}) == {"slice": "4821"}

    def test_random_slice_when_search_slice_id_not_set(self, monkeypatch):
        monkeypatch.setattr(track_module.random, "randint", lambda a, b: 777)
        assert track_module.random_slice_request_params({}) == {"slice": "777"}


def test_bulk_slice_param_source_registered():
    registry = mock.Mock()
    track_module.register(registry)
    registry.register_param_source.assert_any_call("bulk-slice-param-source", track_module.BulkSliceParamSource)
