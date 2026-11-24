"""Device address and size of a loaded GPU kernel's code, by symbol name, through ROCr's loader extension
(each loaded code object kept in memory: its ELF symbol table + the load delta). The layer launch's code
prefetch reads its table (layer.prefetch_code)."""

import ctypes
import struct

_hsa = ctypes.CDLL("libhsa-runtime64.so.1")
_T = None
CB_EXE = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p)
CB_LCO = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_void_p
)


def _table():
    global _T
    if _T is None:
        t = (ctypes.c_void_p * 7)()
        rc = _hsa.hsa_system_get_major_extension_table(
            ctypes.c_uint16(0x201),
            ctypes.c_uint16(1),
            ctypes.c_size_t(ctypes.sizeof(t)),
            t,
        )
        assert rc == 0, rc
        _T = t
    return _T


def _elf_funcs(elf):
    shoff = struct.unpack_from("<Q", elf, 0x28)[0]
    shentsize, shnum, _ = struct.unpack_from("<HHH", elf, 0x3A)
    secs = [
        struct.unpack_from("<IIQQQQIIQQ", elf, shoff + i * shentsize)
        for i in range(shnum)
    ]
    out = {}
    for s in secs:
        if s[1] == 2:  # SHT_SYMTAB
            st = secs[s[6]]
            for j in range(s[5] // 24):
                nm, info, _, _, val, sz = struct.unpack_from(
                    "<IBBHQQ", elf, s[4] + j * 24
                )
                if info & 0xF == 2 and sz:  # STT_FUNC
                    e = elf.index(b"\0", st[4] + nm)
                    out[elf[st[4] + nm : e].decode()] = (val, sz)
    return out


def all_kernels():
    """{name: [(device address, bytes), ...]} over every loaded agent code object of this process."""
    t = _table()
    it_exe = ctypes.CFUNCTYPE(ctypes.c_int, CB_EXE, ctypes.c_void_p)(t[6])
    it_lco = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_uint64, CB_LCO, ctypes.c_void_p)(
        t[3]
    )
    info = ctypes.CFUNCTYPE(
        ctypes.c_int, ctypes.c_uint64, ctypes.c_int, ctypes.c_void_p
    )(t[4])
    found = {}

    def lco_cb(exe, lco, data):
        kind, stype = ctypes.c_uint32(), ctypes.c_uint32()
        base, size, delta = ctypes.c_uint64(), ctypes.c_uint64(), ctypes.c_int64()
        info(lco, 2, ctypes.byref(kind))
        info(lco, 4, ctypes.byref(stype))
        if kind.value != 2 or stype.value != 2:  # agent code objects held in memory
            return 0
        info(lco, 5, ctypes.byref(base))
        info(lco, 6, ctypes.byref(size))
        info(lco, 8, ctypes.byref(delta))
        try:
            elf = ctypes.string_at(base.value, size.value)
            for n, (v, sz) in _elf_funcs(elf).items():
                found.setdefault(n, []).append((v + delta.value, sz))
        except Exception:
            pass
        return 0

    lcb = CB_LCO(lco_cb)

    def exe_cb(exe, data):
        it_lco(exe, lcb, None)
        return 0

    ecb = CB_EXE(exe_cb)
    rc = it_exe(ecb, None)
    assert rc == 0, rc
    return found
