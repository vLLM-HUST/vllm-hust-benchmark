"""Linux namespace helpers for explicitly qualified native code scoring.

Never executes candidate code unless the post-exec isolation checks pass.
This is a bounded local evaluation sandbox, not a security certification.
"""

import ctypes
import os
from pathlib import Path
import stat

SAFE_CHAR_DEVICES = {
    "null": (1, 3),
    "zero": (1, 5),
    "random": (1, 8),
    "urandom": (1, 9),
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def capabilities():
    class Header(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]

    class Data(ctypes.Structure):
        _fields_ = [
            ("effective", ctypes.c_uint32),
            ("permitted", ctypes.c_uint32),
            ("inheritable", ctypes.c_uint32),
        ]

    return Header, Data


def inside(config):
    import resource
    import subprocess

    root, assets = Path(config["root"]), Path(config["assets"])

    def call(*args):
        subprocess.run(args, check=True, stdout=subprocess.DEVNULL)

    call("mount", "--make-rprivate", "/")
    call("mount", "-t", "tmpfs", "-o", "size=64m,nosuid,nodev", "tmpfs", str(root))
    targets = [
        "usr/lib",
        "usr/lib64",
        "proc",
        "tmp",
        "dev/shm",
        "inputs",
        "author",
        "packages",
        "output",
        "sys/devices/system/cpu",
        config["runtime"].lstrip("/"),
    ]
    for name in targets:
        (root / name).mkdir(parents=True, exist_ok=True)

    def bind(source, target, readonly=True):
        call("mount", "--bind", source, str(root / target))
        if readonly:
            call("mount", "-o", "remount,bind,ro,nosuid,nodev", str(root / target))

    for directory in ("/usr/lib", "/usr/lib64"):
        if Path(directory).is_dir():
            bind(directory, directory.lstrip("/"))
    bind(config.get("runtime_source", config["runtime"]), config["runtime"].lstrip("/"))
    bind("/sys/devices/system/cpu", "sys/devices/system/cpu")
    (root / "proc/cpuinfo").touch(exist_ok=False)
    bind("/proc/cpuinfo", "proc/cpuinfo")
    # Runtime-owned executable-discovery shim, NOT a host procfs mount.
    # TensorFlow readlink(/proc/self/exe) needs this path even without procfs.
    (root / "proc/self").mkdir()
    (root / "proc/self/exe").symlink_to(config["runtime"] + "/bin/python3")
    # Match the actual namespace entrypoint argv. This is an explicit runtime
    # compatibility shim, not a claim to expose a general live procfs view.
    entrypoint_argv = [
        config["runtime"] + "/bin/python3",
        "-I",
        "/inputs/worker.py",
        "--evaluate",
    ]
    (root / "proc/self/cmdline").write_bytes(
        b"\0".join(a.encode() for a in entrypoint_argv) + b"\0"
    )
    for directory in ("inputs", "author", "packages"):
        bind(str(assets / directory), directory)
    bind(str(assets / "output"), "output", readonly=False)
    (root / "lib").symlink_to("usr/lib")
    (root / "lib64").symlink_to("usr/lib64")
    call(
        "mount",
        "-t",
        "tmpfs",
        "-o",
        "size=16m,nosuid,nodev",
        "tmpfs",
        str(root / "dev/shm"),
    )
    for name, expected in SAFE_CHAR_DEVICES.items():
        source = Path("/dev") / name
        info = source.stat()
        require(
            stat.S_ISCHR(info.st_mode)
            and (os.major(info.st_rdev), os.minor(info.st_rdev)) == expected,
            "unexpected standard character device",
        )
        target = root / "dev" / name
        target.touch(exist_ok=False)
        call("mount", "--bind", str(source), str(target))
        # nodev would disable these specific harmless character devices.
        # Never bind the whole host /dev or any accelerator/block device.
        call("mount", "-o", "remount,bind,ro,nosuid,noexec", str(target))
    call(
        "mount",
        "-t",
        "tmpfs",
        "-o",
        "size=256m,nosuid,nodev",
        "tmpfs",
        str(root / "tmp"),
    )
    call("mount", "-o", "remount,bind,ro,nosuid,nodev", str(root))
    for name, value in (
        ("RLIMIT_NPROC", 256),
        ("RLIMIT_CPU", 300),
        ("RLIMIT_AS", 2 * 1024**3),
        ("RLIMIT_FSIZE", 16 * 1024**2),
        ("RLIMIT_NOFILE", 128),
    ):
        resource.setrlimit(getattr(resource, name), (value, value))
    require(
        Path("/proc/self/uid_map").read_text().split()
        == ["0", str(config["uid"]), "1"],
        "unexpected host UID mapping",
    )
    os.chdir(root)
    os.chroot(".")
    os.chdir("/")
    libc = ctypes.CDLL(None, use_errno=True)
    require(libc.prctl(38, 1, 0, 0, 0) == 0, "cannot set no_new_privs")
    require(libc.prctl(28, 15, 0, 0, 0) == 0, "cannot lock root capability behavior")
    for cap in range(64):
        result = libc.prctl(24, cap, 0, 0, 0)
        require(
            result == 0 or ctypes.get_errno() == 22,
            "cannot drop capability bounding set",
        )
    header, data = capabilities()
    require(
        libc.capset(ctypes.byref(header(0x20080522, 0)), (data * 2)()) == 0,
        "cannot clear capabilities",
    )
    executable = config["runtime"] + "/bin/python3"
    os.execve(
        executable,
        [executable, "-I", "/inputs/worker.py", "--evaluate"],
        {
            "PATH": config["runtime"] + "/bin",
            "HOME": "/tmp",
            "TMPDIR": "/tmp",
            "LANG": "C.UTF-8",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "TF_NUM_INTRAOP_THREADS": "1",
            "TF_NUM_INTEROP_THREADS": "1",
        },
    )
