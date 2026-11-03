"""Prepare Ubuntu 22.04 runtime libraries locally. Run before starting the editor."""

import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = [
    "libvulkan1",
    "libegl1",
    "libglvnd0",
    "libglx0",
    "libopengl0",
    "libnss3",
    "libnspr4",
    "libatk1.0-0",
    "libatk-bridge2.0-0",
    "libdrm2",
    "libdrm-common",
    "libxcomposite1",
    "libxdamage1",
    "libxfixes3",
    "libxrandr2",
    "libgbm1",
    "libxkbcommon0",
    "libpango-1.0-0",
    "libasound2",
    "libatspi2.0-0",
    "libwayland-server0",
    "libthai0",
    "libdatrie1",
    "libfribidi0",
    "libharfbuzz0b",
    "libgraphite2-3",
    "libdbus-1-3",
    "libxcb-randr0",
]


def main():
    packages = ROOT / ".deps/packages"
    sysroot = ROOT / ".deps/sysroot"
    packages.mkdir(parents=True, exist_ok=True)
    sysroot.mkdir(parents=True, exist_ok=True)

    def missing():
        return [name for name in PACKAGES if not any(packages.glob(name + "_*.deb"))]

    wanted = missing()
    if wanted:
        result = subprocess.run(["apt-get", "download", *wanted], cwd=packages)
        if result.returncode and missing():
            # Refresh into a project-specific index if the host index is stale.
            # Ignore global apt hooks so refreshing does not clean system caches.
            state = ROOT / ".deps/apt"
            (state / "lists/partial").mkdir(parents=True, exist_ok=True)
            (state / "cache/archives/partial").mkdir(parents=True, exist_ok=True)
            source = state / "sources.list"
            source.write_text(
                "deb https://archive.ubuntu.com/ubuntu jammy main\n"
                "deb https://archive.ubuntu.com/ubuntu jammy-updates main\n"
                "deb https://security.ubuntu.com/ubuntu jammy-security main\n"
            )
            options = [
                "-o",
                "Dir::Etc::main=/dev/null",
                "-o",
                "Dir::Etc::parts=-",
                "-o",
                "Dir::Etc::sourcelist=" + str(source),
                "-o",
                "Dir::Etc::sourceparts=-",
                "-o",
                "Dir::State::lists=" + str(state / "lists"),
                "-o",
                "Dir::Cache=" + str(state / "cache"),
            ]
            subprocess.run(["apt-get", *options, "update"], check=True)
            subprocess.run(
                ["apt-get", *options, "download", *missing()], cwd=packages, check=True
            )
    for path in sorted(packages.glob("*.deb")):
        subprocess.run(["dpkg-deb", "-x", str(path), str(sysroot)], check=True)
    receipt = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in packages.glob("*.deb")
    }
    (ROOT / "reports/local_libraries.json").write_text(
        json.dumps(
            dict(
                status="installed_locally",
                system_packages_changed=False,
                packages=receipt,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
