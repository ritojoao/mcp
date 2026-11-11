# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests that bytea, json/jsonb and array columns come back JSON-safe and in one shape."""

import json
import pytest
from awslabs.postgres_mcp_server.connection.psycopg_pool_connection import (
    PsycopgPoolConnection,
)
from awslabs.postgres_mcp_server.server import extract_cell, parse_execute_response
from decimal import Decimal
from unittest.mock import AsyncMock


class _Cursor:
    def __init__(self, columns, rows):
        self.description = [(name,) for name in columns]
        self._rows = rows

    async def execute(self, sql, params=None):
        pass

    async def fetchall(self):
        return self._rows

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _Transaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _Conn:
    def __init__(self, cursor):
        self._cursor = cursor

    def transaction(self):
        return _Transaction()

    def cursor(self):
        return self._cursor

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


async def _run_pg_wire(columns, row):
    """Run execute_query on a mocked psycopg connection and parse like run_query does."""
    connection = PsycopgPoolConnection(
        host='db.example.com',
        port=5432,
        database='test_db',
        readonly=False,
        secret_arn='arn:aws:secretsmanager:us-west-2:1:secret:x',  # pragma: allowlist secret
        db_user='u',
        region='us-west-2',
        is_iam_auth=False,
        is_test=True,
    )
    connection._get_connection = AsyncMock(return_value=_Conn(_Cursor(columns, [row])))
    return parse_execute_response(await connection.execute_query('SELECT 1'))


class TestBytea:
    """bytea is returned as a hex string (PostgreSQL's default bytea_output)."""

    def test_binary_blob_is_hex(self):
        """Bytes that are not valid UTF-8 become a hex string."""
        assert extract_cell({'blobValue': b'\x89PNG\x00\xff'}) == '\\x89504e47' + '00ff'

    def test_utf8_blob_is_still_hex(self):
        """Bytes that happen to be valid UTF-8 are not returned as plain text."""
        assert extract_cell({'blobValue': b'hi'}) == '\\x6869'

    def test_empty_blob(self):
        """An empty bytea is an empty hex string."""
        assert extract_cell({'blobValue': b''}) == '\\x'

    @pytest.mark.asyncio
    async def test_pg_wire_bytea_result_is_json_serializable(self):
        """A bytea column on the PG wire path no longer breaks serialization."""
        rows = await _run_pg_wire(['id', 'payload'], (1, b'\x89PNG\x00\xff'))
        assert json.loads(json.dumps(rows)) == [{'id': 1, 'payload': '\\x89504e4700' + 'ff'}]


class TestArrays:
    """Arrays are returned as plain lists on both paths."""

    def test_rds_api_string_array(self):
        """Data API stringValues unwraps to a list."""
        assert extract_cell({'arrayValue': {'stringValues': ['a', 'b']}}) == ['a', 'b']

    def test_rds_api_number_and_bool_arrays(self):
        """Data API longValues, doubleValues and booleanValues unwrap to lists."""
        assert extract_cell({'arrayValue': {'longValues': [1, 2]}}) == [1, 2]
        assert extract_cell({'arrayValue': {'doubleValues': [1.5]}}) == [1.5]
        assert extract_cell({'arrayValue': {'booleanValues': [True]}}) == [True]

    def test_rds_api_nested_and_empty_arrays(self):
        """Nested arrayValues become nested lists, and an empty array is []."""
        nested = {'arrayValue': {'arrayValues': [{'longValues': [1]}, {'longValues': [2, 3]}]}}
        assert extract_cell(nested) == [[1], [2, 3]]
        assert extract_cell({'arrayValue': {}}) == []

    @pytest.mark.asyncio
    async def test_pg_wire_text_array_is_list(self):
        """A text[] on PG wire is a list, not str(list)."""
        rows = await _run_pg_wire(['tags'], (['red', 'blue'],))
        assert rows == [{'tags': ['red', 'blue']}]

    @pytest.mark.asyncio
    async def test_pg_wire_array_with_null_nested_and_decimal(self):
        """NULL elements, nested arrays and non-JSON elements stay JSON-safe."""
        rows = await _run_pg_wire(['a'], ([1, None, [2, 3], Decimal('1.5')],))
        assert rows == [{'a': [1, None, [2, 3], '1.5']}]
        json.dumps(rows)


class TestJson:
    """json/jsonb is returned as JSON, not as a Python repr."""

    @pytest.mark.asyncio
    async def test_pg_wire_jsonb_object_is_json_text(self):
        """A jsonb object on PG wire matches the JSON text the Data API returns."""
        rows = await _run_pg_wire(['meta'], ({'a': True, 'b': None, 'c': 'x'},))
        assert rows == [{'meta': '{"a": true, "b": null, "c": "x"}'}]
        assert json.loads(rows[0]['meta']) == {'a': True, 'b': None, 'c': 'x'}

    @pytest.mark.asyncio
    async def test_pg_wire_jsonb_array_is_json_list(self):
        """A jsonb array on PG wire is a JSON list."""
        rows = await _run_pg_wire(['meta'], ([1, 'two', {'k': None}],))
        assert rows == [{'meta': [1, 'two', '{"k": null}']}]
