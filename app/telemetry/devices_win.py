"""Windows accelerator discovery through SetupAPI and CfgMgr32.

This module intentionally uses only ``ctypes``.  It does not depend on PowerShell,
WMI, ``pywin32`` or an external monitoring executable.
"""

from __future__ import annotations

import ctypes
import os
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Iterable

from ctypes import wintypes


COMPUTE_ACCELERATOR_CLASS_GUID = "f01a9d53-3ff6-48d2-9f97-c8a7004be10c"
DISPLAY_ADAPTER_CLASS_GUID = "4d36e968-e325-11ce-bfc1-08002be10318"

DEVPKEY_DEVICE_DRIVER_VERSION = "a8b865dd-2e3d-4094-ad97-e593a70c75d6:3"
DEVPKEY_DEVICE_FRIENDLY_NAME = "a45c254e-df1c-4efd-8020-67d146a850e0:14"
DEVPKEY_DEVICE_DEVICE_DESC = "a45c254e-df1c-4efd-8020-67d146a850e0:2"
DEVPKEY_GPU_LUID = "60b193cb-5276-4d0f-96fc-f173abad3ec6:2"
DEVPKEY_GPU_PHYS_ID = "60b193cb-5276-4d0f-96fc-f173abad3ec6:3"

_DIGCF_PRESENT = 0x00000002
_SPDRP_DEVICE_DESC = 0x00000002
_SPDRP_FRIENDLY_NAME = 0x0000000C
_ERROR_INSUFFICIENT_BUFFER = 122
_ERROR_NO_MORE_ITEMS = 259
_INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

_DEVPROP_TYPE_STRING = 0x00000012
_DEVPROP_TYPE_ULONG = 0x00000003
_DEVPROP_TYPE_UINT64 = 0x0000000B
_DEVPROP_TYPE_MASK = 0x0000FFFF

_NPU_NAME_RE = re.compile(r"\b(?:npu|neural|ai\s*boost)\b", re.IGNORECASE)


class GUID(ctypes.Structure):
    _fields_ = (
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", ctypes.c_ubyte * 8),
    )

    @classmethod
    def parse(cls, value: str) -> "GUID":
        parts = value.split("-")
        if len(parts) != 5:
            raise ValueError(f"invalid GUID: {value!r}")
        # GUID Data1/Data2/Data3 are serialized little-endian in memory even
        # though their canonical string representation is big-endian.
        raw = int(parts[0], 16).to_bytes(4, "little")
        raw += int(parts[1], 16).to_bytes(2, "little")
        raw += int(parts[2], 16).to_bytes(2, "little")
        raw += bytes((int(parts[3][0:2], 16), int(parts[3][2:4], 16)))
        raw += bytes.fromhex(parts[4])
        return cls.from_buffer_copy(raw)


class DEVPROPKEY(ctypes.Structure):
    _fields_ = (
        ("fmtid", GUID),
        ("pid", wintypes.DWORD),
    )


class SP_DEVINFO_DATA(ctypes.Structure):
    _fields_ = (
        ("cbSize", wintypes.DWORD),
        ("ClassGuid", GUID),
        ("DevInst", wintypes.DWORD),
        ("Reserved", wintypes.PULONG),
    )


@dataclass(frozen=True)
class DeviceInfo:
    device_id: str
    friendly_name: str
    description: str
    driver_version: str
    phys_id: int | None
    luid: int | None
    class_guid: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class WindowsApiError(RuntimeError):
    """A Windows API call failed in a way that prevents device discovery."""


def _devpropkey(value: str) -> DEVPROPKEY:
    guid_text, property_id = value.split(":", 1)
    return DEVPROPKEY(GUID.parse(guid_text), int(property_id))


@lru_cache(maxsize=1)
def _windows_api() -> tuple[object, object]:
    if os.name != "nt":
        raise WindowsApiError("Windows SetupAPI is only available on Windows")

    setupapi = ctypes.WinDLL("setupapi.dll", use_last_error=True)
    cfgmgr32 = ctypes.WinDLL("cfgmgr32.dll", use_last_error=True)

    setupapi.SetupDiGetClassDevsW.argtypes = (
        ctypes.POINTER(GUID),
        wintypes.LPCWSTR,
        wintypes.HWND,
        wintypes.DWORD,
    )
    setupapi.SetupDiGetClassDevsW.restype = wintypes.HANDLE

    setupapi.SetupDiEnumDeviceInfo.argtypes = (
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(SP_DEVINFO_DATA),
    )
    setupapi.SetupDiEnumDeviceInfo.restype = wintypes.BOOL

    setupapi.SetupDiGetDeviceInstanceIdW.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(SP_DEVINFO_DATA),
        wintypes.LPWSTR,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    )
    setupapi.SetupDiGetDeviceInstanceIdW.restype = wintypes.BOOL

    setupapi.SetupDiGetDeviceRegistryPropertyW.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(SP_DEVINFO_DATA),
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    )
    setupapi.SetupDiGetDeviceRegistryPropertyW.restype = wintypes.BOOL

    setupapi.SetupDiGetDevicePropertyW.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(SP_DEVINFO_DATA),
        ctypes.POINTER(DEVPROPKEY),
        ctypes.POINTER(wintypes.DWORD),
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.DWORD,
    )
    setupapi.SetupDiGetDevicePropertyW.restype = wintypes.BOOL

    setupapi.SetupDiDestroyDeviceInfoList.argtypes = (wintypes.HANDLE,)
    setupapi.SetupDiDestroyDeviceInfoList.restype = wintypes.BOOL

    # Keep a CfgMgr32 symbol in the wrapper and expose a devnode status helper for
    # operator diagnostics. Device enumeration itself uses SetupAPI instance IDs.
    cfgmgr32.CM_Get_DevNode_Status.argtypes = (wintypes.DWORD, ctypes.POINTER(wintypes.ULONG))
    cfgmgr32.CM_Get_DevNode_Status.restype = wintypes.DWORD

    return setupapi, cfgmgr32


def devnode_status(devnode: int) -> tuple[int, int]:
    """Return ``(result, status)`` from ``CM_Get_DevNode_Status``."""

    setupapi, cfgmgr32 = _windows_api()
    del setupapi
    status = wintypes.ULONG()
    result = int(cfgmgr32.CM_Get_DevNode_Status(wintypes.DWORD(devnode), ctypes.byref(status)))
    return result, int(status.value)


def _get_instance_id(setupapi: object, device_info_set: int, info: SP_DEVINFO_DATA) -> str:
    required = wintypes.DWORD()
    ctypes.set_last_error(0)
    ok = setupapi.SetupDiGetDeviceInstanceIdW(
        device_info_set,
        ctypes.byref(info),
        None,
        0,
        ctypes.byref(required),
    )
    if ok or ctypes.get_last_error() != _ERROR_INSUFFICIENT_BUFFER:
        raise ctypes.WinError(ctypes.get_last_error())
    buffer = ctypes.create_unicode_buffer(required.value)
    if not setupapi.SetupDiGetDeviceInstanceIdW(
        device_info_set,
        ctypes.byref(info),
        buffer,
        required.value,
        None,
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return buffer.value


def _get_registry_string(
    setupapi: object,
    device_info_set: int,
    info: SP_DEVINFO_DATA,
    property_id: int,
) -> str:
    required = wintypes.DWORD()
    ctypes.set_last_error(0)
    setupapi.SetupDiGetDeviceRegistryPropertyW(
        device_info_set,
        ctypes.byref(info),
        property_id,
        None,
        None,
        0,
        ctypes.byref(required),
    )
    if not required.value:
        return ""
    data_type = wintypes.DWORD()
    buffer = ctypes.create_string_buffer(required.value)
    if not setupapi.SetupDiGetDeviceRegistryPropertyW(
        device_info_set,
        ctypes.byref(info),
        property_id,
        ctypes.byref(data_type),
        buffer,
        required.value,
        None,
    ):
        return ""
    return buffer.value.decode("utf-16-le", errors="replace").rstrip("\x00")


def _get_property(
    setupapi: object,
    device_info_set: int,
    info: SP_DEVINFO_DATA,
    property_key: str,
    value_kind: str,
) -> tuple[int | None, bytes | str | None]:
    key = _devpropkey(property_key)
    required = wintypes.DWORD()
    property_type = wintypes.DWORD()
    ctypes.set_last_error(0)
    setupapi.SetupDiGetDevicePropertyW(
        device_info_set,
        ctypes.byref(info),
        ctypes.byref(key),
        ctypes.byref(property_type),
        None,
        0,
        ctypes.byref(required),
        0,
    )
    if not required.value:
        return None, None
    buffer = ctypes.create_string_buffer(required.value)
    if not setupapi.SetupDiGetDevicePropertyW(
        device_info_set,
        ctypes.byref(info),
        ctypes.byref(key),
        ctypes.byref(property_type),
        buffer,
        required.value,
        ctypes.byref(required),
        0,
    ):
        return None, None
    raw = buffer.raw[: required.value]
    base_type = int(property_type.value) & _DEVPROP_TYPE_MASK
    if value_kind == "string" and base_type == _DEVPROP_TYPE_STRING:
        return None, raw.decode("utf-16-le", errors="replace").rstrip("\x00")
    if value_kind == "uint32" and len(raw) >= 4:
        return int.from_bytes(raw[:4], "little"), None
    if value_kind == "uint64" and len(raw) >= 8:
        return int.from_bytes(raw[:8], "little"), None
    return None, None


def enumerate_class(class_guid_text: str) -> list[DeviceInfo]:
    """Enumerate present devices in a SetupAPI device class."""

    setupapi, _ = _windows_api()
    class_guid = GUID.parse(class_guid_text)
    device_info_set = setupapi.SetupDiGetClassDevsW(
        ctypes.byref(class_guid),
        None,
        None,
        _DIGCF_PRESENT,
    )
    if device_info_set in (None, 0, _INVALID_HANDLE_VALUE):
        raise ctypes.WinError(ctypes.get_last_error())

    devices: list[DeviceInfo] = []
    try:
        index = 0
        while True:
            info = SP_DEVINFO_DATA()
            info.cbSize = ctypes.sizeof(SP_DEVINFO_DATA)
            ctypes.set_last_error(0)
            if not setupapi.SetupDiEnumDeviceInfo(device_info_set, index, ctypes.byref(info)):
                error = ctypes.get_last_error()
                if error == _ERROR_NO_MORE_ITEMS:
                    break
                raise ctypes.WinError(error)
            index += 1
            try:
                instance_id = _get_instance_id(setupapi, device_info_set, info)
                friendly = _get_property(
                    setupapi, device_info_set, info, DEVPKEY_DEVICE_FRIENDLY_NAME, "string"
                )[1]
                if not isinstance(friendly, str) or not friendly:
                    friendly = _get_registry_string(setupapi, device_info_set, info, _SPDRP_FRIENDLY_NAME)
                description = _get_property(
                    setupapi, device_info_set, info, DEVPKEY_DEVICE_DEVICE_DESC, "string"
                )[1]
                if not isinstance(description, str) or not description:
                    description = _get_registry_string(setupapi, device_info_set, info, _SPDRP_DEVICE_DESC)
                if not friendly:
                    friendly = description
                driver = _get_property(
                    setupapi, device_info_set, info, DEVPKEY_DEVICE_DRIVER_VERSION, "string"
                )[1]
                if not isinstance(driver, str) or not driver:
                    driver = ""
                phys_id, _ = _get_property(
                    setupapi, device_info_set, info, DEVPKEY_GPU_PHYS_ID, "uint32"
                )
                luid, _ = _get_property(
                    setupapi, device_info_set, info, DEVPKEY_GPU_LUID, "uint64"
                )
                devices.append(
                    DeviceInfo(
                        device_id=instance_id,
                        friendly_name=friendly,
                        description=description,
                        driver_version=driver,
                        phys_id=phys_id,
                        luid=luid,
                        class_guid=class_guid_text,
                    )
                )
            except OSError:
                # A device can disappear between enumeration and property reads.
                continue
    finally:
        setupapi.SetupDiDestroyDeviceInfoList(device_info_set)
    return devices


def enumerate_compute_accelerators() -> list[DeviceInfo]:
    return enumerate_class(COMPUTE_ACCELERATOR_CLASS_GUID)


def enumerate_display_adapters() -> list[DeviceInfo]:
    return enumerate_class(DISPLAY_ADAPTER_CLASS_GUID)


def is_npu(device: DeviceInfo) -> bool:
    text = f"{device.friendly_name} {device.description} {device.device_id}"
    return bool(_NPU_NAME_RE.search(text))


def is_intel(device: DeviceInfo) -> bool:
    text = f"{device.friendly_name} {device.description} {device.device_id}".lower()
    return "intel" in text or "ven_8086" in text.lower()


def driver_tuple(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for part in version.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            break
    return tuple(parts)


def driver_at_least(version: str, minimum: str) -> bool:
    actual = driver_tuple(version)
    required = driver_tuple(minimum)
    width = max(len(actual), len(required))
    return actual + (0,) * (width - len(actual)) >= required + (0,) * (width - len(required))


def select_driver(devices: Iterable[DeviceInfo], *, npu: bool) -> DeviceInfo | None:
    candidates = [device for device in devices if is_intel(device) and (is_npu(device) if npu else True)]
    if not candidates:
        candidates = [device for device in devices if is_npu(device)] if npu else []
    if not candidates:
        return None
    return max(candidates, key=lambda item: driver_tuple(item.driver_version))
