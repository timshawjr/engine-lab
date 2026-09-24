"""Minimal ``pdh.dll`` wrapper for Windows GPU Engine counters.

The public surface deliberately returns individual counter samples.  It does not
invent a system utilization number by averaging unrelated engines or processes.
"""

from __future__ import annotations

import ctypes
import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from ctypes import wintypes


_PDH_MORE_DATA = 0x800007D2
_PDH_FMT_DOUBLE = 0x00000200
_PDH_FMT_NOCAP100 = 0x00008000
_PERF_DETAIL_WIZARD = 0x00000004
_ERROR_SUCCESS = 0

_GPU_ENGINE_OBJECT = r"\GPU Engine"
_UTILIZATION_COUNTER = "Utilization Percentage"
_RUNNING_TIME_COUNTER = "Running time"

_INSTANCE_RE = re.compile(
    r"^pid_(?P<pid>\d+)_luid_0x(?P<luid_device>[0-9A-Fa-f]+)_0x(?P<luid_node>[0-9A-Fa-f]+)"
    r"_phys_(?P<phys>\d+)_eng_(?P<engine>\d+)_engtype_(?P<engtype>.*)$"
)


class PDHError(RuntimeError):
    """A Performance Data Helper call failed."""


class _CounterValueUnion(ctypes.Union):
    _fields_ = (
        ("long_value", wintypes.LONG),
        ("double_value", ctypes.c_double),
        ("large_value", ctypes.c_longlong),
        ("ansi_string_value", ctypes.c_char_p),
        ("wide_string_value", ctypes.c_wchar_p),
    )


class _PDH_FMT_COUNTERVALUE(ctypes.Structure):
    _fields_ = (
        ("status", wintypes.DWORD),
        ("value", _CounterValueUnion),
    )


class _PDH_FMT_COUNTERVALUE_ITEM_W(ctypes.Structure):
    _fields_ = (
        ("name", ctypes.c_wchar_p),
        ("formatted_value", _PDH_FMT_COUNTERVALUE),
    )


class _PDH_COUNTER_PATH_ELEMENTS_W(ctypes.Structure):
    _fields_ = (
        ("machine_name", ctypes.c_wchar_p),
        ("object_name", ctypes.c_wchar_p),
        ("instance_name", ctypes.c_wchar_p),
        ("parent_instance", ctypes.c_wchar_p),
        ("instance_index", wintypes.DWORD),
        ("counter_name", ctypes.c_wchar_p),
    )


class _PDH_COUNTER_INFO_W(ctypes.Structure):
    _fields_ = (
        ("length", wintypes.DWORD),
        ("counter_type", wintypes.DWORD),
        ("version", wintypes.DWORD),
        ("status", wintypes.DWORD),
        ("scale", wintypes.LONG),
        ("default_scale", wintypes.LONG),
        ("user_data", ctypes.c_size_t),
        ("query_user_data", ctypes.c_size_t),
        ("full_path", ctypes.c_wchar_p),
        ("counter_path", _PDH_COUNTER_PATH_ELEMENTS_W),
        ("explain_text", ctypes.c_wchar_p),
        ("data_buffer", wintypes.DWORD * 1),
    )


@dataclass(frozen=True)
class ParsedInstance:
    instance_name: str
    pid: int | None
    phys_id: int | None
    engine_id: int | None
    engtype: str | None
    luid_device: int | None
    luid_node: int | None

    @property
    def luid(self) -> str | None:
        if self.luid_device is None or self.luid_node is None:
            return None
        return f"0x{self.luid_device:08x}_0x{self.luid_node:08x}"


@dataclass(frozen=True)
class CounterSample:
    instance: ParsedInstance
    value: float
    counter_name: str
    measured_at: float


def _unsigned_status(value: int) -> int:
    return ctypes.c_ulong(value).value


def _require_success(status: int, operation: str, *, allow_more_data: bool = False) -> int:
    code = _unsigned_status(status)
    if code == _ERROR_SUCCESS or (allow_more_data and code == _PDH_MORE_DATA):
        return code
    raise PDHError(f"{operation} failed with PDH status 0x{code:08X}")


@lru_cache(maxsize=1)
def _pdh_api() -> object:
    if os.name != "nt":
        raise PDHError("pdh.dll is only available on Windows")
    dll = ctypes.WinDLL("pdh.dll", use_last_error=True)

    dll.PdhOpenQueryW.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE))
    dll.PdhOpenQueryW.restype = wintypes.LONG

    dll.PdhCloseQuery.argtypes = (wintypes.HANDLE,)
    dll.PdhCloseQuery.restype = wintypes.LONG

    dll.PdhAddEnglishCounterW.argtypes = (
        wintypes.HANDLE,
        wintypes.LPCWSTR,
        ctypes.c_size_t,
        ctypes.POINTER(wintypes.HANDLE),
    )
    dll.PdhAddEnglishCounterW.restype = wintypes.LONG

    dll.PdhCollectQueryData.argtypes = (wintypes.HANDLE,)
    dll.PdhCollectQueryData.restype = wintypes.LONG

    dll.PdhGetFormattedCounterArrayW.argtypes = (
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.POINTER(wintypes.DWORD),
        ctypes.c_void_p,
    )
    dll.PdhGetFormattedCounterArrayW.restype = wintypes.LONG

    dll.PdhGetCounterInfoW.argtypes = (
        wintypes.HANDLE,
        wintypes.BOOL,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.c_void_p,
    )
    dll.PdhGetCounterInfoW.restype = wintypes.LONG

    dll.PdhEnumObjectItemsW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        ctypes.c_wchar_p,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.c_wchar_p,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.DWORD,
        wintypes.DWORD,
    )
    dll.PdhEnumObjectItemsW.restype = wintypes.LONG
    return dll


def _split_double_nul(buffer: ctypes.Array, char_count: int) -> list[str]:
    if char_count <= 0:
        return []
    raw = ctypes.string_at(buffer, char_count)
    text = raw.decode("utf-16-le", errors="replace")
    values = [value for value in text.split("\x00") if value]
    return values


def _localized_gpu_engine_object_name() -> str:
    """Resolve the localized object name through a language-neutral counter."""

    dll = _pdh_api()
    query = wintypes.HANDLE()
    status = dll.PdhOpenQueryW(None, 0, ctypes.byref(query))
    _require_success(status, "PdhOpenQueryW for object localization")
    counter = wintypes.HANDLE()
    try:
        status = dll.PdhAddEnglishCounterW(
            query,
            rf"{_GPU_ENGINE_OBJECT}(*)\{_UTILIZATION_COUNTER}",
            0,
            ctypes.byref(counter),
        )
        _require_success(status, "PdhAddEnglishCounterW for object localization")
        required = wintypes.DWORD()
        status = dll.PdhGetCounterInfoW(counter, False, ctypes.byref(required), None)
        _require_success(status, "PdhGetCounterInfoW size query", allow_more_data=True)
        if not required.value:
            raise PDHError("PdhGetCounterInfoW returned an empty structure size")
        raw = ctypes.create_string_buffer(required.value)
        info = ctypes.cast(raw, ctypes.POINTER(_PDH_COUNTER_INFO_W))
        status = dll.PdhGetCounterInfoW(
            counter,
            False,
            ctypes.byref(required),
            ctypes.byref(info[0]),
        )
        _require_success(status, "PdhGetCounterInfoW")
        object_name = info[0].counter_path.object_name
        if not object_name:
            raise PDHError("localized GPU Engine object name is empty")
        return str(object_name)
    finally:
        status = dll.PdhCloseQuery(query)
        _require_success(status, "PdhCloseQuery after object localization")


def enumerate_gpu_engine_items() -> tuple[list[str], list[str]]:
    """Return counter and instance names for the local ``GPU Engine`` object."""

    dll = _pdh_api()
    object_name = _localized_gpu_engine_object_name()
    counter_size = wintypes.DWORD()
    instance_size = wintypes.DWORD()
    status = dll.PdhEnumObjectItemsW(
        None,
        None,
        object_name,
        None,
        ctypes.byref(counter_size),
        None,
        ctypes.byref(instance_size),
        _PERF_DETAIL_WIZARD,
        0,
    )
    _require_success(status, "PdhEnumObjectItemsW size query", allow_more_data=True)
    if not counter_size.value and not instance_size.value:
        return [], []

    counter_buffer = ctypes.create_unicode_buffer(max(counter_size.value, 1))
    instance_buffer = ctypes.create_unicode_buffer(max(instance_size.value, 1))
    status = dll.PdhEnumObjectItemsW(
        None,
        None,
        object_name,
        counter_buffer,
        ctypes.byref(counter_size),
        instance_buffer,
        ctypes.byref(instance_size),
        _PERF_DETAIL_WIZARD,
        0,
    )
    _require_success(status, "PdhEnumObjectItemsW")
    counters = _split_double_nul(counter_buffer, counter_size.value)
    instances = _split_double_nul(instance_buffer, instance_size.value)
    return counters, instances


def parse_instance(instance_name: str) -> ParsedInstance:
    match = _INSTANCE_RE.match(instance_name)
    if match is None:
        return ParsedInstance(instance_name, None, None, None, None, None, None)
    return ParsedInstance(
        instance_name=instance_name,
        pid=int(match.group("pid")),
        phys_id=int(match.group("phys")),
        engine_id=int(match.group("engine")),
        engtype=match.group("engtype"),
        luid_device=int(match.group("luid_device"), 16),
        luid_node=int(match.group("luid_node"), 16),
    )


def classify_device(
    instance: ParsedInstance,
    npu_phys_ids: Iterable[int],
    npu_luids: Iterable[int] = (),
    gpu_luids: Iterable[int] = (),
    gpu_phys_ids: Iterable[int] = (),
) -> str:
    """Classify an instance, using LUID first when physical ids collide.

    Lunar Lake exposes phys_0 for both the NPU and the integrated GPU.  LUID is
    therefore the authoritative tie-breaker on this machine; physical-id-only
    classification would incorrectly label every GPU instance as NPU.
    """

    npu_luid_set = set(npu_luids)
    gpu_luid_set = set(gpu_luids)
    if instance.luid_node is not None and instance.luid_node in npu_luid_set:
        return "NPU"
    if instance.luid_node is not None and instance.luid_node in gpu_luid_set:
        return "GPU"
    if instance.phys_id is None:
        return "UNKNOWN"
    phys_id = instance.phys_id
    npu_phys_set = set(npu_phys_ids)
    gpu_phys_set = set(gpu_phys_ids)
    if phys_id in npu_phys_set and phys_id not in gpu_phys_set:
        return "NPU"
    if phys_id in gpu_phys_set and phys_id not in npu_phys_set:
        return "GPU"
    return "UNKNOWN"


class GPUEngineCounter:
    """Sampling handle for wildcard ``GPU Engine`` performance counters."""

    def __init__(self) -> None:
        self._dll = _pdh_api()
        self._query = wintypes.HANDLE()
        status = self._dll.PdhOpenQueryW(None, 0, ctypes.byref(self._query))
        _require_success(status, "PdhOpenQueryW")
        self._utilization = self._add_counter(_UTILIZATION_COUNTER)
        self._running_time = self._add_counter(_RUNNING_TIME_COUNTER)
        self._running_previous: dict[str, float] = {}
        self._running_collected_at: float | None = None
        self._closed = False

    def _add_counter(self, counter_name: str) -> wintypes.HANDLE:
        handle = wintypes.HANDLE()
        path = rf"{_GPU_ENGINE_OBJECT}(*)\{counter_name}"
        status = self._dll.PdhAddEnglishCounterW(
            self._query,
            path,
            0,
            ctypes.byref(handle),
        )
        _require_success(status, f"PdhAddEnglishCounterW({counter_name})")
        return handle

    def __enter__(self) -> "GPUEngineCounter":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            status = self._dll.PdhCloseQuery(self._query)
            _require_success(status, "PdhCloseQuery")
            self._closed = True

    def collect(self) -> None:
        status = self._dll.PdhCollectQueryData(self._query)
        _require_success(status, "PdhCollectQueryData")

    def _formatted_samples(self, counter: wintypes.HANDLE, counter_name: str) -> list[CounterSample]:
        buffer_size = wintypes.DWORD()
        item_count = wintypes.DWORD()
        status = self._dll.PdhGetFormattedCounterArrayW(
            counter,
            _PDH_FMT_DOUBLE | _PDH_FMT_NOCAP100,
            ctypes.byref(buffer_size),
            ctypes.byref(item_count),
            None,
        )
        _require_success(status, f"PdhGetFormattedCounterArrayW({counter_name})", allow_more_data=True)
        if not buffer_size.value:
            return []

        raw_buffer = ctypes.create_string_buffer(buffer_size.value)
        items = ctypes.cast(raw_buffer, ctypes.POINTER(_PDH_FMT_COUNTERVALUE_ITEM_W))
        status = self._dll.PdhGetFormattedCounterArrayW(
            counter,
            _PDH_FMT_DOUBLE | _PDH_FMT_NOCAP100,
            ctypes.byref(buffer_size),
            ctypes.byref(item_count),
            ctypes.byref(items[0]),
        )
        _require_success(status, f"PdhGetFormattedCounterArrayW({counter_name}) refill")
        measured_at = time.time()
        samples: list[CounterSample] = []
        for index in range(item_count.value):
            item = items[index]
            if not item.name or item.formatted_value.status != 0:
                continue
            instance = parse_instance(item.name)
            samples.append(
                CounterSample(
                    instance=instance,
                    value=float(item.formatted_value.value.double_value),
                    counter_name=counter_name,
                    measured_at=measured_at,
                )
            )
        return samples

    def sample_utilization(self) -> list[CounterSample]:
        self.collect()
        return self._formatted_samples(self._utilization, _UTILIZATION_COUNTER)

    def sample_running_time_delta(self) -> list[CounterSample]:
        """Return per-instance running-time delta as a percentage.

        This is a delta-based fallback only.  The method deliberately does not
        combine instances; the caller can choose and label an aggregation.
        """

        self.collect()
        now = time.monotonic()
        samples = self._formatted_samples(self._running_time, _RUNNING_TIME_COUNTER)
        previous = self._running_previous
        previous_at = self._running_collected_at
        elapsed = None if previous_at is None else max(now - previous_at, 0.0)
        current = {sample.instance.instance_name: sample.value for sample in samples}
        result: list[CounterSample] = []
        if elapsed is not None and elapsed > 0:
            for sample in samples:
                old = previous.get(sample.instance.instance_name)
                if old is None or sample.value < old:
                    continue
                result.append(
                    CounterSample(
                        instance=sample.instance,
                        value=(sample.value - old) / elapsed * 100.0,
                        counter_name=f"{_RUNNING_TIME_COUNTER} (delta %)",
                        measured_at=sample.measured_at,
                    )
                )
        self._running_previous = current
        self._running_collected_at = now
        return result

    def prime_running_time(self, interval_s: float = 1.0) -> None:
        time.sleep(interval_s)
        self.sample_running_time_delta()
