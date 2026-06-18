# Copyright 2025 The Torch-Spyre Authors.
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

"""Regression tests for ``_find_allocation``'s bisect cache lifecycle.

The cache (``_FIND_ALLOC_CACHE``) speeds up byte-addr → allocation lookups,
but must not leak: it should evict entries when a memory store is cleared and
when a store is garbage-collected (so repeated benchmark/test runs that each
build a fresh store don't grow it without bound).
"""

import gc

import numpy as np

from ktir_cpu.memory import (
    HBMSimulator,
    LXScratchpad,
    _AllocStore,
    _FIND_ALLOC_CACHE,
    _find_allocation,
    _read_flat,
)


def test_clear_evicts_cache_entry():
    """LXScratchpad.clear() must drop the store's cache entry."""
    lx = LXScratchpad(size_mb=1, core_id=0)
    lx.write(0, np.arange(16, dtype=np.float16))
    # Touch an *interior* address (byte offset 4) so the lookup goes through the
    # bisect path and populates the cache — reading exactly at a base_ptr hits
    # the `ptr in memory` fast path and never caches.
    lx.read(4, 8, "f16")
    assert lx.memory in _FIND_ALLOC_CACHE

    lx.clear()
    assert lx.memory not in _FIND_ALLOC_CACHE, "clear() left a stale cache entry"


def test_gc_of_store_auto_evicts():
    """A garbage-collected store must not pin its cache entry (weak keys)."""
    hbm = HBMSimulator(size_gb=1)
    stick = hbm.allocate(32)
    hbm.write(stick, np.arange(16, dtype=np.float16))
    # Interior read (intra_byte=4) → bisect path → cache entry populated.
    hbm.read(stick, 8, "f16", intra_byte=4)

    store = hbm.memory
    assert store in _FIND_ALLOC_CACHE
    before = len(_FIND_ALLOC_CACHE)

    # Drop every strong reference to the store and collect.
    del hbm, store
    gc.collect()

    assert len(_FIND_ALLOC_CACHE) < before, (
        "cache did not shrink after the store was collected — entries leak "
        "across repeated runs"
    )


def test_cache_rebuilds_on_new_allocation():
    """Adding an allocation (len change) invalidates and the lookup stays correct."""
    lx = LXScratchpad(size_mb=1, core_id=0)
    lx.write(0, np.arange(8, dtype=np.float16))
    np.testing.assert_array_equal(lx.read(0, 8, "f16"), np.arange(8, dtype=np.float16))

    # Second, higher allocation: the cache must rebuild its sorted-key list and
    # still resolve an address that falls inside the new allocation.
    lx.write(64, np.arange(100, 108, dtype=np.float16))
    np.testing.assert_array_equal(
        lx.read(64, 8, "f16"), np.arange(100, 108, dtype=np.float16)
    )
    # An address inside the first allocation still resolves correctly.
    np.testing.assert_array_equal(lx.read(0, 8, "f16"), np.arange(8, dtype=np.float16))


def test_plain_dict_bypasses_cache():
    """A plain dict (not an _AllocStore) is looked up without touching the cache."""
    mem = {0x1000: np.arange(16, dtype=np.float16)}
    before = len(_FIND_ALLOC_CACHE)

    flat = _read_flat(mem, 0x1004, 13, np.float16, 2)
    np.testing.assert_array_equal(flat, np.arange(2, 15, dtype=np.float16))

    # Plain dicts are unhashable / not weak-referenceable; they must be served
    # without ever being inserted into the WeakKeyDictionary.
    assert len(_FIND_ALLOC_CACHE) == before
    assert _find_allocation(mem, 0x1004, 2) == (0x1000, mem[0x1000], 2)


def test_alloc_store_is_identity_keyed():
    """_AllocStore hashes/compares by identity so equal-content stores are distinct keys."""
    a = _AllocStore()
    b = _AllocStore()
    a[0] = np.zeros(1, dtype=np.float16)
    b[0] = np.zeros(1, dtype=np.float16)
    assert a == a and not (a == b)  # identity, not content equality
    assert hash(a) != hash(b) or a is b
