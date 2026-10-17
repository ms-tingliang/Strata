"""Whether a Windows shared expert arena (--shared-expert-arena NAME / STRATA_SHARED_EXPERT_ARENA) still exists, and
what its header says: the state (loading / ready), the engine that loaded it, its size, pack hash and page size.

    python tools/shared_arena_status.py NAME

Exit code 0: no section of that name in this Windows session (every engine that used it has ended and Windows freed
it); 1: it exists; 2: this is not Windows, or it could not be asked.  Only the Python standard library (ctypes); the
script maps the header page alone and closes its handle before it ends, so it never keeps the section alive.
"""
import ctypes
import struct
import sys

FILE_MAP_READ = 0x0004
ERROR_FILE_NOT_FOUND = 2
HEADER = struct.Struct("<16sIIQQQQQQ")   # magic, version, header_bytes, arena_bytes, pack_hash, state, loader_pid, flags, -
STATES = {0: "not written", 1: "loading", 2: "ready"}


def section_name(name: str) -> str:
    """The engine's rule: a NAME starting with Local\\ or Global\\ as given, else Local\\NAME with \\ and control
    characters as _."""
    if name.startswith(("Local\\", "Global\\")):
        return name
    return "Local\\" + "".join("_" if c == "\\" or ord(c) < 32 else c for c in name)


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] in ("-h", "--help"):
        print(__doc__.strip())
        return 2
    if sys.platform != "win32":
        print("shared_arena_status: Windows only (on Linux the shared arena is the file given to --shared-expert-arena)")
        return 2
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenFileMappingW.restype = ctypes.c_void_p
    k32.OpenFileMappingW.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_wchar_p]
    k32.MapViewOfFile.restype = ctypes.c_void_p
    k32.MapViewOfFile.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_size_t]
    k32.UnmapViewOfFile.argtypes = [ctypes.c_void_p]
    k32.CloseHandle.argtypes = [ctypes.c_void_p]
    name = section_name(argv[1])
    h = k32.OpenFileMappingW(FILE_MAP_READ, 0, name)
    if not h:
        err = ctypes.get_last_error()
        if err == ERROR_FILE_NOT_FOUND:
            print(f"{name}: no such section - freed (or never created in this Windows session)")
            return 0
        print(f"{name}: cannot be asked (OpenFileMappingW error {err})")
        return 2
    try:
        v = k32.MapViewOfFile(h, FILE_MAP_READ, 0, 0, HEADER.size)
        if not v:
            print(f"{name}: EXISTS (its header cannot be mapped: error {ctypes.get_last_error()})")
            return 1
        try:
            magic, version, _hb, arena, pack, state, pid, flags, _ = HEADER.unpack(ctypes.string_at(v, HEADER.size))
        finally:
            k32.UnmapViewOfFile(v)
    finally:
        k32.CloseHandle(h)
    if magic.rstrip(b"\0") != b"STRATA-ARENA-V1" or version != 1:
        print(f"{name}: EXISTS, but it is not a Strata expert arena")
        return 1
    print(f"{name}: EXISTS - {STATES.get(state, f'state {state}')}, loaded by engine PID {pid}, "
          f"{arena / 2**30:.2f} GiB of experts, pack hash {pack:016x}, {'large' if flags & 1 else '4 KB'} pages")
    print("  it lives while an engine (or another program) has it open; Windows frees it when the last one ends")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
